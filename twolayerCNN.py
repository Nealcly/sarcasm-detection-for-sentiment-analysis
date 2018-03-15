#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 1/18/2018 7:05 PM
# @Author  : Leyang
import tensorflow as tf
import numpy as np


class twolayerCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
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
        pooled_outputs_1 = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-1-%s" % filter_size):
                # 1st Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs_1.append(pooled)



        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool_1 = tf.concat(pooled_outputs_1, 3)
        #self.h_pool_flat_1 = tf.reshape(self.h_pool_1, [-1, num_filters_total])

        #reshape(batch, width, height and channel.)
        self.h_pool_flat_1 = tf.reshape(self.h_pool_1, [-1, num_filters_total, 1, 1])

        #2nd convolution layer
        pooled_outputs_2 = []
        for i ,filter_size in enumerate(filter_sizes):
            with tf.name_scope("con-maxpool-2-%s" %filter_size):
                filter_shape = [filter_size, 1, 1, num_filters]
                #input and output channel are 1 and num_filters
                W = tf.Variable(tf.truncated_normal(filter_shape,stddev = 0.1), name = "W")
                b = tf.Variable(tf.constant(0.1,shape = [num_filters]) , name = "b")
                conv = tf.nn.conv2d(
                    self.h_pool_flat_1,
                    W,
                    strides = [1,1,1,1],
                    padding = "VALID",
                    name = "conv"
                )
                #nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv , b) , name ="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs_2.append(pooled)


        #Combine all the pooled features
        self.h_pool_2 = tf.concat(pooled_outputs_1, 3)
        self.h_pool_flat_2 = tf.reshape(self.h_pool_2, [-1, num_filters_total])




        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat_2, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
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



