'''
Created on 27 Sep 2017

'''

import tensorflow as tf
import numpy as np
from attention import attention_mechanism
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn
from tensorflow.contrib.rnn import GRUCell

class attentionBiLSTM(object):
    """
    Attention BiLSTM model
    """

    def __init__(
        self, input_embedding_size, sequence_length, hidden_size, output_size, vocab_size, learning_rate ,ATTENTION_SIZE):

        #Placeholder for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None,sequence_length], name ="train_input")
        self.input_y = tf.placeholder(tf.float32, [None,output_size], name="train_output")
        #self.train_input_embedding = tf.placeholder(tf.float32, [None,sequence_length,input_embedding_size], name="train_input_embedding")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        ################
        #Embedding layer
        ################
        with tf.device('/cpu:0'), tf.name_scope("Embedding"):
            self.W = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -0.0001, 0.0001),trainable=True, name="W", dtype = tf.float32)
            self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, input_embedding_size], name="embedding_placeholder")
            self.embedding_init = self.W.assign(self.embedding_placeholder)
            self.train_input_embedding = tf.nn.embedding_lookup(self.W, self.input_x)
    

        ###########
        #attention LSTM model
        ###########
        # Define weights
        with tf.name_scope("Model"):
            weights = tf.Variable(tf.random_normal([2*hidden_size, output_size]))
            biases = tf.Variable(tf.random_normal([output_size]))
            #GRUCell/BasicLSTMCell

            lstm_output, _ = bidirectional_dynamic_rnn(GRUCell(hidden_size),
                                                       GRUCell(hidden_size),
                                                     inputs=self.train_input_embedding,dtype=tf.float32)

            # Attention layer
            outputs, self.alphas = attention_mechanism.attention(lstm_output, ATTENTION_SIZE, return_alphas=True)
            #outputs, self.alphas = attention_mechanism.attention(lstm_output, ATTENTION_SIZE, return_alphas=True)
            # Add dropout
            outputs = tf.nn.dropout(outputs, self.dropout_keep_prob)




            # Get lstm cell output
            #outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)


            # Linear activation, using rnn inner loop last output
            self.scores = tf.matmul(outputs, weights) + biases
            #self.scores = tf.matmul(outputs[-1], weights) + biases
            self.pred_ops = tf.nn.softmax(self.scores)
            self.predictions = tf.argmax(self.scores, 1, name="predictions")


        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits = self.scores, labels = self.input_y)
            self.loss = tf.reduce_mean(losses)# + l2_reg_lambda * l2_loss
#             optimizer = tf.train.AdamOptimizer(learning_rate)
#             grads_and_vars = optimizer.compute_gradients(self.loss)
# #             global_step = tf.Variable(0, name="global_step", trainable=False)
#             self.train_op = optimizer.apply_gradients(grads_and_vars)
  
        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            self.y = tf.argmax(self.input_y, 1)


    