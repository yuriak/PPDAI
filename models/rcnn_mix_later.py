import itertools
import tensorflow as tf
from models.layers import add_dense_layer, dot_attention, biGRUs, add_GRU, add_CNNs
from models.base_model import BaseModel


class RCnnQNetMixLater(BaseModel):
    def __init__(self, word_embedding, char_embedding, learning_rate=0.0001, log_dir='./logs'):
        super().__init__(word_embedding, char_embedding, 'RCnn_QNet_MixLater', log_dir=log_dir)

        self._build_model(word_embedding, char_embedding, rnn_encoder_units_number=[256, 128],
                          rnn_decoder_units_number=[32], attention_size=[128],
                          attention_cnn_size=[128, 64], kernel_size=2, learning_rate=learning_rate)
        self.init_op = tf.global_variables_initializer()
        self.merge_op = tf.summary.merge_all()
        self.saver = tf.train.Saver()
        self.session = tf.Session()
        self.writer = tf.summary.FileWriter(log_dir,
                                            graph=self.session.graph
                                            )
        self.session.run(self.init_op)

    def _build_model(self, word_embedding, char_embedding, rnn_encoder_units_number, rnn_decoder_units_number,
                     attention_size,
                     attention_cnn_size, kernel_size, learning_rate=0.0001):
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
            fcell, bcell = biGRUs(rnn_encoder_units_number, activation=tf.nn.relu, keep_prob=self.dropout_keep_prob)
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
            fcell, bcell = biGRUs(rnn_encoder_units_number, activation=tf.nn.relu, keep_prob=self.dropout_keep_prob)
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

        with tf.variable_scope('q1_attention', initializer=tf.contrib.layers.xavier_initializer(uniform=True),reuse=tf.AUTO_REUSE) as scope:
            self.q1wc_att = dot_attention(self.q1we,self.q1ce, hidden=attention_size, scope='q1wc_attention')
            self.q1wc_att = add_CNNs(inputs=self.q1wc_att, hidden_units=attention_cnn_size, kernel_size=kernel_size, activation=tf.nn.relu, keep_prob=self.dropout_keep_prob)
            #
            self.q1cw_att = dot_attention(self.q1ce,self.q1we, hidden=attention_size, scope='q1cw_attention')
            self.q1cw_att = add_CNNs(inputs=self.q1cw_att, hidden_units=attention_cnn_size, kernel_size=kernel_size, activation=tf.nn.relu, keep_prob=self.dropout_keep_prob)
            tf.summary.histogram('q1wc_att', self.q1wc_att)
            tf.summary.histogram('q1cw_att', self.q1cw_att)

        with tf.variable_scope('q2_attention', initializer=tf.contrib.layers.xavier_initializer(uniform=True),reuse=tf.AUTO_REUSE) as scope:
            self.q2wc_att = dot_attention(self.q2we,self.q2ce, hidden=attention_size, scope='q2wc_attention')
            self.q2wc_att = add_CNNs(inputs=self.q2wc_att, hidden_units=attention_cnn_size, kernel_size=kernel_size, activation=tf.nn.relu, keep_prob=self.dropout_keep_prob)
            #
            self.q2cw_att = dot_attention(self.q2ce,self.q2we, hidden=attention_size, scope='q2cw_attention')
            self.q2cw_att = add_CNNs(inputs=self.q2cw_att, hidden_units=attention_cnn_size, kernel_size=kernel_size, activation=tf.nn.relu, keep_prob=self.dropout_keep_prob)
            tf.summary.histogram('q2wc_att', self.q2wc_att)
            tf.summary.histogram('q2cw_att', self.q2cw_att)

        with tf.variable_scope('wc_output_layer',
                               initializer=tf.contrib.layers.xavier_initializer(uniform=True)) as scope:
            fcell, bcell = biGRUs(rnn_decoder_units_number, activation=tf.nn.relu, keep_prob=self.dropout_keep_prob)

            self.wc = dot_attention(self.q1wc_att, self.q2wc_att, hidden=attention_size, scope='wc_attention')
            self._, self.wc = tf.nn.bidirectional_dynamic_rnn(inputs=self.wc, cell_fw=fcell, cell_bw=bcell,
                                                              dtype=tf.float32, scope='wc_decoder')
            self.wc = list(itertools.chain.from_iterable(self.wc))
            self.wc = tf.concat(self.wc, axis=-1)

        with tf.variable_scope('cw_output_layer',
                               initializer=tf.contrib.layers.xavier_initializer(uniform=True)) as scope:
            fcell, bcell = biGRUs(rnn_decoder_units_number, activation=tf.nn.relu, keep_prob=self.dropout_keep_prob)
            self.cw = dot_attention(self.q1cw_att, self.q2cw_att, hidden=attention_size, scope='cw_attention')
            self._, self.cw = tf.nn.bidirectional_dynamic_rnn(inputs=self.cw, cell_fw=fcell, cell_bw=bcell,
                                                              dtype=tf.float32, scope='cw_decoder')
            self.cw = list(itertools.chain.from_iterable(self.cw))
            self.cw = tf.concat(self.cw, axis=-1)

        with tf.variable_scope('output_layer', initializer=tf.contrib.layers.xavier_initializer(uniform=True)) as scope:
            self.o = tf.concat([self.wc, self.cw], axis=-1)
            # self.o=tf.concat([self.s3,self.s4],axis=-1)
            self.o = tf.contrib.layers.layer_norm(self.o, scope=scope)
            self.o = add_dense_layer(self.o, [128, 64, 2], self.dropout_keep_prob, activation=tf.nn.relu, use_bias=True)
            self.output = tf.nn.softmax(self.o)
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.o, labels=self.y))

        with tf.variable_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            tf.summary.scalar('loss', self.loss)
            self.train_op = self.optimizer.minimize(self.loss)
