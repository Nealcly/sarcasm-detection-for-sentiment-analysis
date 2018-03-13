import tensorflow as tf
import numpy as np


class GatedCNN_nopadding(object):
    """
    Uses an embedding layer, followed by a convolutional,gated, and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda,learning_rate):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)



        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -0.25, 0.25),trainable=True,
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)


            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size

        filter_size = filter_sizes[0]
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            # Convolution Layer
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            W1 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W1")
            b1 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b1")
            conv = tf.nn.conv2d(
                self.embedded_chars_expanded,
                W1,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")

            h1 = tf.add(conv, b1)

            W2 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W2")
            b2 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b2")
            conv = tf.nn.conv2d(
                self.embedded_chars_expanded,
                W2,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")
            h2 = tf.add(conv, b2)

            #add forget gate
            h = h1 * tf.sigmoid(h2)
            print (h.shape)
            h = tf.reshape(h, (-1, (num_filters * (sequence_length - filter_size + 1))))
            print (h.shape)



        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(h, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[h.get_shape()[1], num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
            # optimizer = tf.train.AdamOptimizer(learning_rate)
            # grads_and_vars = optimizer.compute_gradients(self.loss)
            # self.train_op = optimizer.apply_gradients(grads_and_vars)


        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            self.y = tf.argmax(self.input_y, 1)



