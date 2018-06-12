import tensorflow as tf


def add_dense_layer(inputs, output_shape, drop_keep_prob, act=tf.nn.tanh, use_bias=True):
    output = inputs
    for n in output_shape:
        output = tf.layers.dense(output, n, activation=act, use_bias=use_bias)
        output = tf.nn.dropout(output, drop_keep_prob)
    return output


def dot_attention(inputs, memory, hidden, keep_prob=1.0, is_train=None, scope="dot_attention",
                  self_attention=False):
    with tf.variable_scope(scope):
        with tf.variable_scope("attention"):
            inputs_ = add_dense_layer(inputs, hidden, keep_prob, act=tf.nn.relu, use_bias=False)
            memory_ = add_dense_layer(memory, hidden, keep_prob, act=tf.nn.relu, use_bias=False)
            outputs = tf.matmul(inputs_, tf.transpose(memory_, [0, 2, 1]))
            logits = tf.nn.softmax(outputs)
            outputs = tf.matmul(logits, memory)
            result = tf.concat([inputs, outputs], axis=-1)
        with tf.variable_scope("gate"):
            gate = add_dense_layer(result, [result.shape[-1]], keep_prob, act=tf.nn.sigmoid, use_bias=False)
            return result * gate


def biGRUs(units_number, activation=tf.nn.tanh, keep_prob=1.0):
    fcell = [add_GRU(units_number=n, keep_prob=keep_prob, activation=activation) for n in units_number]
    fcell = tf.contrib.rnn.MultiRNNCell(cells=fcell, state_is_tuple=True)
    bcell = [add_GRU(units_number=n, keep_prob=keep_prob, activation=activation) for n in units_number]
    bcell = tf.contrib.rnn.MultiRNNCell(cells=bcell, state_is_tuple=True)
    return fcell, bcell


def add_GRU(units_number, activation=tf.nn.tanh, keep_prob=1.0):
    cell = tf.contrib.rnn.GRUCell(units_number, activation=activation)
    cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob)
    return cell
