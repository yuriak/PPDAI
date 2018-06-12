import tensorflow as tf
import os
from layers import add_dense_layer, dot_attention, biGRUs, add_GRU


class BaseModel(object):

    def __init__(self, word_embedding, char_embedding, model_name, log_dir):
        self.word_embedding = word_embedding
        self.char_embedding = char_embedding
        self._name = model_name
        tf.reset_default_graph()
        self.q1_words = tf.placeholder(shape=[None, None], dtype=tf.int32, name='q1_words')
        self.q2_words = tf.placeholder(shape=[None, None], dtype=tf.int32, name='q2_words')
        self.q1_chars = tf.placeholder(shape=[None, None], dtype=tf.int32, name='q1_chars')
        self.q2_chars = tf.placeholder(shape=[None, None], dtype=tf.int32, name='q2_chars')
        self.y = tf.placeholder(shape=[None], dtype=tf.int32, name='y_start')
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_keep_prob')
        self.global_step = 0

        self.init_op = tf.global_variables_initializer()
        self.merge_op = tf.summary.merge_all()
        self.session = tf.Session()
        self.session.run(self.init_op)
        self.saver = tf.train.Saver()
        self.writer = tf.summary.FileWriter(log_dir,
                                            graph=self.session.graph
                                            )

        self.train_op = None
        self.loss = None
        self.output = None

    def train(self, q1w, q1c, q2w, q2c, y, drop_keep_prob=0.85, record_interval=10):
        feed_dict = {
            self.q1_words: q1w,
            self.q2_words: q2w,
            self.q1_chars: q1c,
            self.q2_chars: q2c,
            self.y: y,
            self.dropout_keep_prob: drop_keep_prob
        }
        if self.global_step % record_interval == 0:
            _, loss, summaries = self.session.run([self.train_op, self.loss, self.merge_op], feed_dict=feed_dict)
            self.writer.add_summary(summaries, self.global_step)
        else:
            _, loss = self.session.run([self.train_op, self.loss], feed_dict=feed_dict)
        self.global_step += 1
        return loss

    def evaluate(self, q1w, q1c, q2w, q2c, y, drop_keep_prob=1.0):
        feed_dict = {
            self.q1_words: q1w,
            self.q2_words: q2w,
            self.q1_chars: q1c,
            self.q2_chars: q2c,
            self.y: y,
            self.dropout_keep_prob: drop_keep_prob
        }
        loss = self.session.run([self.loss], feed_dict=feed_dict)[0]
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = loss
        summary_value.tag = 'evaluate_loss'
        self.writer.add_summary(summary, self.global_step)
        return loss

    def predict(self, q1w, q1c, q2w, q2c):
        feed_dict = {
            self.q1_words: q1w,
            self.q2_words: q2w,
            self.q1_chars: q1c,
            self.q2_chars: q2c,
            self.dropout_keep_prob: 1.0
        }
        y_hat = self.session.run([self.output], feed_dict=feed_dict)[0]
        return y_hat[:, 1]

    def load_model(self, model_path='./QModel'):
        self.saver.restore(self.session, model_path + '/rnnqnet')

    def save_model(self, model_path='./QModel'):
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        model_file = model_path + '/rnnqnet'
        self.saver.save(self.session, model_file)


class RnnQNet(BaseModel):
    def __init__(self, word_embedding, char_embedding, log_dir='./logs'):
        super().__init__(word_embedding, char_embedding, 'Rnn_QNet', log_dir=log_dir)

        self._build_model(word_embedding, char_embedding, encoder_units_number=[512, 256], attention_size=[128],
                          hidden_rnn_size=[128], learning_rate=0.0001, )

    def _build_model(self, word_embedding, char_embedding, encoder_units_number, attention_size,
                     hidden_rnn_size, learning_rate=0.0001):

        with tf.variable_scope('word_embedding', initializer=tf.contrib.layers.xavier_initializer()):
            Ww = tf.Variable(word_embedding, trainable=True, dtype=tf.float32)
            self.q1w = tf.nn.embedding_lookup(ids=self.q1_words, params=Ww)
            self.q2w = tf.nn.embedding_lookup(ids=self.q2_words, params=Ww)

        with tf.variable_scope('char_embedding', initializer=tf.contrib.layers.xavier_initializer()):
            Wc = tf.Variable(char_embedding, trainable=True, dtype=tf.float32)
            self.q1c = tf.nn.embedding_lookup(ids=self.q1_chars, params=Wc)
            self.q2c = tf.nn.embedding_lookup(ids=self.q2_chars, params=Wc)

        with tf.variable_scope('word_encoder', initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                               reuse=tf.AUTO_REUSE) as scope:
            fcell, bcell = biGRUs(encoder_units_number, activation=tf.nn.relu, keep_prob=self.dropout_keep_prob)
            self.q1we, self._ = tf.nn.bidirectional_dynamic_rnn(inputs=self.q1w, cell_fw=fcell, cell_bw=bcell,
                                                                dtype=tf.float32, scope='q1w_encoder')
            self.q1we = tf.concat(self.q1we, axis=-1)
            self.q1we = tf.contrib.layers.layer_norm(self.q1we, scope=scope)

            tf.get_variable_scope().reuse_variables()
            self.q2we, self._ = tf.nn.bidirectional_dynamic_rnn(inputs=self.q2w, cell_fw=fcell, cell_bw=bcell,
                                                                dtype=tf.float32, scope='q2w_encoder')
            self.q2we = tf.concat(self.q2we, axis=-1)

            self.q2we = tf.contrib.layers.layer_norm(self.q2we, scope=scope)
            tf.summary.histogram('q1we', self.q1we)
            tf.summary.histogram('q2we', self.q2we)

        with tf.variable_scope('char_encoder', initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                               reuse=tf.AUTO_REUSE) as scope:
            fcell, bcell = biGRUs(encoder_units_number, activation=tf.nn.relu, keep_prob=self.dropout_keep_prob)
            self.q1ce, self._ = tf.nn.bidirectional_dynamic_rnn(inputs=self.q1c, cell_fw=fcell, cell_bw=bcell,
                                                                dtype=tf.float32, scope='q1c_encoder')
            self.q1ce = tf.concat(self.q1ce, axis=-1)
            self.q1ce = tf.contrib.layers.layer_norm(self.q1ce, scope=scope)

            tf.get_variable_scope().reuse_variables()
            self.q2ce, self._ = tf.nn.bidirectional_dynamic_rnn(inputs=self.q2c, cell_fw=fcell, cell_bw=bcell,
                                                                dtype=tf.float32, scope='q2c_encoder')
            self.q2ce = tf.concat(self.q2ce, axis=-1)

            self.q2ce = tf.contrib.layers.layer_norm(self.q2ce, scope=scope)
            tf.summary.histogram('q1ce', self.q1ce)
            tf.summary.histogram('q2ce', self.q2ce)

        with tf.variable_scope('word_attention', initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                               reuse=tf.AUTO_REUSE) as scope:
            fcell, bcell = biGRUs(hidden_rnn_size, activation=tf.nn.relu, keep_prob=self.dropout_keep_prob)

            self.q1q2w_att = dot_attention(self.q1we, self.q2we, hidden=attention_size, scope='q1q2w_attention')
            self.q1q2w_att, _ = tf.nn.bidirectional_dynamic_rnn(inputs=self.q1q2w_att, cell_fw=fcell, cell_bw=bcell,
                                                                dtype=tf.float32, scope='q1q2w_attention_rnn')
            self.q1q2w_att = tf.concat(self.q1q2w_att, axis=-1)
            self.q1q2w_att = tf.contrib.layers.layer_norm(self.q1q2w_att, scope=scope)

            self.q2q1w_att = dot_attention(self.q2we, self.q1we, hidden=attention_size, scope='q2q1w_attention')
            tf.get_variable_scope().reuse_variables()
            self.q2q1w_att, _ = tf.nn.bidirectional_dynamic_rnn(inputs=self.q2q1w_att, cell_fw=fcell, cell_bw=bcell,
                                                                dtype=tf.float32, scope='q2q1w_attention_rnn')
            self.q2q1w_att = tf.concat(self.q2q1w_att, axis=-1)
            self.q2q1w_att = tf.contrib.layers.layer_norm(self.q2q1w_att, scope=scope)
            tf.summary.histogram('q1q2_att', self.q1q2w_att)
            tf.summary.histogram('q2q1w_att', self.q2q1w_att)

        with tf.variable_scope('char_attention', initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                               reuse=tf.AUTO_REUSE) as scope:
            fcell, bcell = biGRUs(hidden_rnn_size, activation=tf.nn.relu, keep_prob=self.dropout_keep_prob)

            self.q1q2c_att = dot_attention(self.q1ce, self.q2ce, hidden=attention_size, scope='q1q2c_attention')
            self.q1q2c_att, _ = tf.nn.bidirectional_dynamic_rnn(inputs=self.q1q2c_att, cell_fw=fcell, cell_bw=bcell,
                                                                dtype=tf.float32, scope='q1q2c_attention_rnn')
            self.q1q2c_att = tf.concat(self.q1q2c_att, axis=-1)
            self.q1q2c_att = tf.contrib.layers.layer_norm(self.q1q2c_att, scope=scope)

            self.q2q1c_att = dot_attention(self.q2ce, self.q1ce, hidden=attention_size, scope='q2q1c_attention')
            tf.get_variable_scope().reuse_variables()
            self.q2q1c_att, _ = tf.nn.bidirectional_dynamic_rnn(inputs=self.q2q1c_att, cell_fw=fcell, cell_bw=bcell,
                                                                dtype=tf.float32, scope='q2q1c_attention_rnn')
            self.q2q1c_att = tf.concat(self.q2q1c_att, axis=-1)
            self.q2q1c_att = tf.contrib.layers.layer_norm(self.q2q1c_att, scope=scope)
            tf.summary.histogram('q1q2c_att', self.q1q2c_att)
            tf.summary.histogram('q2q1c_att', self.q2q1c_att)

        with tf.variable_scope('word_output_layer',
                               initializer=tf.contrib.layers.xavier_initializer(uniform=True)) as scope:
            wgru = add_GRU(hidden_rnn_size[-1] // 2, activation=tf.nn.relu, keep_prob=self.dropout_keep_prob)
            _, self.s1 = tf.nn.dynamic_rnn(inputs=self.q1q2w_att, cell=wgru, dtype=tf.float32, scope='s1')
            tf.get_variable_scope().reuse_variables()
            _, self.s2 = tf.nn.dynamic_rnn(inputs=self.q2q1w_att, cell=wgru, dtype=tf.float32, scope='s2')
            # # s1_number=tf.reduce_sum(tf.abs(tf.sign(self.s1)),axis=1)
            # # s2_number=tf.reduce_sum(tf.abs(tf.sign(self.s2)),axis=1)
            # s1_number=tf.count_nonzero(self.s1,axis=1,dtype=tf.float32)
            # s2_number=tf.count_nonzero(self.s2,axis=1,dtype=tf.float32)
            # self.s1=tf.reduce_sum(self.s1,axis=1)/s1_number
            # self.s2=tf.reduce_sum(self.s2,axis=1)/s2_number
            # self.s1=tf.reduce_mean(self.s1,axis=1)
            # self.s2=tf.reduce_mean(self.s2,axis=1)

        with tf.variable_scope('char_output_layer',
                               initializer=tf.contrib.layers.xavier_initializer(uniform=True)) as scope:
            cgru = add_GRU(hidden_rnn_size[-1] // 2, activation=tf.nn.relu, keep_prob=self.dropout_keep_prob)
            _, self.s3 = tf.nn.dynamic_rnn(inputs=self.q1q2c_att, cell=cgru, dtype=tf.float32, scope='s3')
            tf.get_variable_scope().reuse_variables()
            _, self.s4 = tf.nn.dynamic_rnn(inputs=self.q2q1c_att, cell=cgru, dtype=tf.float32, scope='s4')
            # # s3_number=tf.reduce_sum(tf.abs(tf.sign(self.s3)),axis=1)
            # # s4_number=tf.reduce_sum(tf.abs(tf.sign(self.s4)),axis=1)
            # s3_number=tf.count_nonzero(self.s3,axis=1,dtype=tf.float32)
            # s4_number=tf.count_nonzero(self.s4,axis=1,dtype=tf.float32)
            # self.s3=tf.reduce_sum(self.s3,axis=1)/s3_number
            # self.s4=tf.reduce_sum(self.s4,axis=1)/s4_number
            # self.s3=tf.reduce_mean(self.s3,axis=1)
            # self.s4=tf.reduce_mean(self.s4,axis=1)

        with tf.variable_scope('output_layer', initializer=tf.contrib.layers.xavier_initializer(uniform=True)) as scope:
            self.o = tf.concat([self.s1, self.s2, self.s3, self.s4], axis=-1)
            # self.o=tf.concat([self.s3,self.s4],axis=-1)
            self.o = tf.contrib.layers.layer_norm(self.o, scope=scope)
            self.o = add_dense_layer(self.o, [128, 64, 2], self.dropout_keep_prob, act=tf.nn.relu, use_bias=True)
            self.output = tf.nn.softmax(self.o)
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.o, labels=self.y))

        with tf.variable_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            tf.summary.scalar('loss', self.loss)
            self.train_op = self.optimizer.minimize(self.loss)
