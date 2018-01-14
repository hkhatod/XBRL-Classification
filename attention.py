import tensorflow as tf


def attention(inputs, rnn_cell_size, hidden_layer_size, sequence_length):
    X = tf.reshape(inputs, [-1, 2*rnn_cell_size])
    Y = tf.layers.dense(X, hidden_layer_size, activation=tf.nn.relu)
    logits =  tf.layers.dense(Y, 1, activation=None)

    logits = tf.reshape(logits, [-1, sequence_length, 1])
    alphas = tf.nn.softmax(logits, dim=1)
    encoded_sentence = inputs * alphas
    return encoded_sentence, alphas
