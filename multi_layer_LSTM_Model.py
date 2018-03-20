import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

class multi_layer_LSTM(object):
    """
    multi_layer_LSTM model
    """
    
    def __init__(
        self, input_embedding_size, sequence_length, hidden_size, output_size, vocab_size, learning_rate, num_layers):

        #Placeholder for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None,sequence_length], name ="train_input")
        self.input_y = tf.placeholder(tf.float32, [None,output_size], name="train_output")
        #self.train_input_embedding = tf.placeholder(tf.float32, [None,sequence_length,input_embedding_size], name="train_input_embedding")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        ################
        #Embedding layer
        ################
        with tf.device('/cpu:0'), tf.name_scope("Embedding"):
            self.W = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -0.1, 0.1),trainable=True, name="W", dtype = tf.float32)
            self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, input_embedding_size], name="embedding_placeholder")
            self.embedding_init = self.W.assign(self.embedding_placeholder)
            self.train_input_embedding = tf.nn.embedding_lookup(self.W, self.input_x)
    

        ###########
        #LSTM model
        ###########
        # Define weights
        with tf.name_scope("Model"):
            weights = tf.Variable(tf.random_normal([hidden_size, output_size]))
            biases = tf.Variable(tf.random_normal([output_size]))
            


            # Prepare data shape to match `rnn` function requirements
            # Current data input shape: (batch_size, n_steps, n_input)
            # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
        
            # Permuting batch_size and n_steps
            x = tf.transpose(self.train_input_embedding, [1, 0, 2])
            # Reshaping to (n_steps*batch_size, n_input)
            x = tf.reshape(x, [-1, input_embedding_size])
            # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
            x = tf.split(x, sequence_length, 0)


            mlstm_cell = []
            for i in range(num_layers):
                # Define a lstm cell with tensorflow
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0)
                # Add dropout
                lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.dropout_keep_prob)
                mlstm_cell.append(lstm_cell)
            mlstm_cell = rnn.MultiRNNCell(cells = mlstm_cell)

            # Get lstm cell output
            outputs, states = tf.contrib.rnn.static_rnn(mlstm_cell, x, dtype=tf.float32)

            # Linear activation, using rnn inner loop mean output
            self.scores = tf.matmul(sum(outputs)/len(outputs), weights) + biases
            self.pred_ops = tf.nn.softmax(self.scores)
            self.predictions = tf.argmax(self.scores, 1, name="predictions")


        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits = self.scores, labels = self.input_y)
            self.loss = tf.reduce_mean(losses)# + l2_reg_lambda * l2_loss
            optimizer = tf.train.AdamOptimizer(learning_rate)
            grads_and_vars = optimizer.compute_gradients(self.loss)
#             global_step = tf.Variable(0, name="global_step", trainable=False)
            self.train_op = optimizer.apply_gradients(grads_and_vars)
  
        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            self.y = tf.argmax(self.input_y, 1)