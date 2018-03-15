import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers

from sklearn.metrics import precision_score, recall_score, f1_score
from LSTM_Model import LSTM
from Bi_LSTM_Model import BiLSTM
from Attention_LSTM_Model import attentionLSTM
from tensorflow.contrib import learn
from gensim.models.keyedvectors import KeyedVectors
import sklearn.metrics
import time
import csv

from multi_layer_LSTM_Model import multi_layer_LSTM
from Self_Attention_BiLSTM_Model import self_attention_BiLSTM



np.set_printoptions(threshold=np.inf)
# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_float("test_sample_percentage", .2, "Percentage of the training data to use for test")
tf.flags.DEFINE_string("train_sarcasm", "./dataset/train/train_sarcasm.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("train_nonsarcasm", "./dataset/train/train_nonsarcasm.txt", "Data source for the negative data.")
tf.flags.DEFINE_string("dev_test_sarcasm", "./dataset/dev_test/dev_test_sarcasm.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("dev_test_nonsarcasm", "./dataset/dev_test/dev_test_nonsarcasm.txt", "Data source for the negative data.")
#_split
tf.flags.DEFINE_string("positive_data_file", "./dataset/train/train_sarcasm.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./dataset/train/train_nonsarcasm.txt", "Data source for the negative data.")
#Model
tf.flags.DEFINE_string("model","highway_layer_cnn", "[highway_layer_cnn,cnn,gate_cnn,gate_cnn_nopadding,twolayerCNN,twolayerCNNnopooling,lstm,bilstm,attention-bilstm,muliti-layer-lstm,muliti-layer-Bilstm]")
#embedding
tf.flags.DEFINE_string("embedding","word2vec", "word2vec,glove")
# Model Hyperparameters
#tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 1.0 , "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 0.0001, "learning_rate")
#CNN
tf.flags.DEFINE_integer("embedding_dim", 300, "Pretrain Word2vec (default: 300)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
#LSTM
tf.flags.DEFINE_integer("hidden_sizes", 128, "Number of hidden sizes (default: 128)")
#num_layers
tf.flags.DEFINE_integer("num_layers", 2, "Number of hidden layers(default: 2)")

#self_attention
tf.flags.DEFINE_integer("d_a",100,"d_a")
tf.flags.DEFINE_integer("r",2,"how many different parts to be extracted from the sentence")
#tf.flags.DEFINE_integer("attention_size", 300, "ATTENTION_SIZE")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

x_text_train, y_train = data_helpers.load_data_and_labels(FLAGS.train_sarcasm, FLAGS.train_nonsarcasm)
x_text_dev_test, y_dev_test = data_helpers.load_data_and_labels(FLAGS.dev_test_sarcasm, FLAGS.dev_test_nonsarcasm)
x_text = x_text_train + x_text_dev_test

max_document_length = max([len(x.split(" ")) for x in x_text])

print ("max_document_length",max_document_length)
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
with open ("x_train.txt","w",encoding='utf-8') as f:
    for i in x_text_train:
        f.write(i + "\n")
with open ("x_dev_test.txt","w",encoding='utf-8') as f:
    for i in x_text_dev_test:
        f.write(i + "\n")



x_train = np.array(list(vocab_processor.fit_transform(x_text_train)))
print (x_train.shape)
x_dev_test = np.array(list(vocab_processor.fit_transform(x_text_dev_test)))
y_original = y_dev_test


# Randomly shuffle data
np.random.seed(10)
shuffle_indices_train = np.random.permutation(np.arange(len(y_train)))
np.random.seed(10)
shuffle_indices_dev_test = np.random.permutation(np.arange(len(y_dev_test)))

x_train = x_train[shuffle_indices_train]
y_train = y_train[shuffle_indices_train]
x_dev_test = x_dev_test[shuffle_indices_dev_test]
y_dev_test = y_dev_test[shuffle_indices_dev_test]
x_dev ,y_dev = x_dev_test[:600],y_dev_test[:600]
x_test ,y_test = x_dev_test[600:],y_dev_test[600:]

print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev/Test split: {:d}/{:d}/{:d}".format(len(y_train), len(y_dev) ,len(y_test)))
########combine





# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        np.random.seed(1)
        tf.set_random_seed(2)
        if FLAGS.model == "cnn":
            from text_cnn import TextCNN
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=len(y_train[1]),
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                learning_rate = FLAGS.learning_rate)
        elif FLAGS.model == "gate_cnn":
            from gated_cnn import GatedCNN
            cnn = GatedCNN(
                sequence_length=x_train.shape[1],
                num_classes=len(y_train[1]),
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                learning_rate = FLAGS.learning_rate)
        elif FLAGS.model == "gate_cnn_nopadding":
            from gated_cnn_nopadding import GatedCNN_nopadding
            cnn = GatedCNN_nopadding(
                sequence_length=x_train.shape[1],
                num_classes=len(y_train[1]),
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                learning_rate = FLAGS.learning_rate)
        elif FLAGS.model == "twolayerCNN_nopooling":
            from twolayerCNN_nopooling import twolayerCNN_no_pooling
            cnn = twolayerCNN_no_pooling(
                sequence_length=x_train.shape[1],
                num_classes=len(y_train[1]),
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                learning_rate = FLAGS.learning_rate)
        elif FLAGS.model == "twolayerCNN":
            from twolayerCNN import twolayerCNN
            cnn = twolayerCNN(
                sequence_length=x_train.shape[1],
                num_classes=len(y_train[1]),
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                learning_rate = FLAGS.learning_rate)
        elif FLAGS.model == "muliti-layer-Bilstm":
            from multi_layer_Bi_LSTM_Model import multi_layer_BiLSTM
            cnn = multi_layer_BiLSTM(
                 input_embedding_size = FLAGS.embedding_dim,
                 sequence_length = x_train.shape[1],
                 #hidden_size = FLAGS.num_filters * len(list(map(int, FLAGS.filter_sizes.split(",")))),
                 hidden_size=FLAGS.hidden_sizes,
                 output_size = y_train.shape[1],
                 vocab_size = len(vocab_processor.vocabulary_),
                 learning_rate = FLAGS.learning_rate,
                 num_layers = FLAGS.num_layers
            )
        elif FLAGS.model == "muliti-layer-lstm":
            cnn = multi_layer_LSTM(
                 input_embedding_size = FLAGS.embedding_dim,
                 sequence_length = x_train.shape[1],
                 #hidden_size = FLAGS.num_filters * len(list(map(int, FLAGS.filter_sizes.split(",")))),
                 hidden_size=FLAGS.hidden_sizes,
                 output_size = y_train.shape[1],
                 vocab_size = len(vocab_processor.vocabulary_),
                 learning_rate = FLAGS.learning_rate,
                 num_layers = FLAGS.num_layers)
        elif FLAGS.model == "attention-bilstm":
            from Attention_BiLSTM_Model import attentionBiLSTM
            cnn = attentionBiLSTM(
                 input_embedding_size = FLAGS.embedding_dim,
                 sequence_length = x_train.shape[1],
                 #hidden_size = FLAGS.num_filters * len(list(map(int, FLAGS.filter_sizes.split(",")))),
                 hidden_size=FLAGS.hidden_sizes,
                 output_size = y_train.shape[1],
                 vocab_size = len(vocab_processor.vocabulary_),
                 learning_rate = FLAGS.learning_rate,
                 ATTENTION_SIZE = max_document_length)
        elif FLAGS.model == "self_attention_BiLSTM":
            cnn = self_attention_BiLSTM(
                input_embedding_size=FLAGS.embedding_dim,
                sequence_length=x_train.shape[1],
                # hidden_size = FLAGS.num_filters * len(list(map(int, FLAGS.filter_sizes.split(",")))),
                hidden_size=FLAGS.hidden_sizes,
                output_size=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                learning_rate=FLAGS.learning_rate,
                d_a = FLAGS.d_a,
                r = FLAGS.r)
        elif FLAGS.model == "highway_layer_cnn":
            from highway_layer_cnn import hightway_CNN
            cnn = hightway_CNN(
                sequence_length=x_train.shape[1],
                num_classes=len(y_train[1]),
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                learning_rate = FLAGS.learning_rate)



        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        #optimizer = tf.train.MomentumOptimizer(FLAGS.learning_rate,0.99)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        #load pre-train word2vec 300d
        print("Start Loading Embedding!")
        if FLAGS.embedding == "word2vec":
            word2vec = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
        #load pre-train glove 200d
        elif FLAGS.embedding == "glove":
            word2vec = KeyedVectors.load_word2vec_format('glove.twitter.27B.200d.bin', binary=True)
        print("Finish Loading Embedding!")
        my_embedding_matrix = np.zeros(shape=(len(vocab_processor.vocabulary_), FLAGS.embedding_dim))
        for word in vocab_processor.vocabulary_._mapping:
            id = vocab_processor.vocabulary_._mapping[word]
            if word in word2vec.vocab:
                my_embedding_matrix[id] = word2vec[word]
            else:
                my_embedding_matrix[id] = np.random.uniform(low=-0.0001, high=0.0001, size=FLAGS.embedding_dim)
        W = tf.placeholder(tf.float32, [None, None], name="pretrained_embeddings")
        set_x = cnn.W.assign(my_embedding_matrix)
        sess.run(set_x, feed_dict={W: my_embedding_matrix})
        print("Finish transfer")

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy, predictions,y_actual = sess.run(
                     [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy, cnn.predictions,cnn.y],
                     feed_dict)
            time_str = datetime.datetime.now().isoformat()
            # print("train_f1_score:", f1_score(y_actual, predictions, average=None))
            # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            return accuracy

            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }

            step, summaries, loss, accuracy ,predictions,y_actual= sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.predictions,cnn.y],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            #print ("predictions",predictions)
            f1 = f1_score(y_actual, predictions, average=None)
            precision = precision_score(y_actual, predictions, average=None)
            recall = recall_score(y_actual, predictions, average=None)
            dev_result = {
                "dev_f1_score_sarcasm": f1[1],
                "dev_precision_sarcasm": precision[1],
                "dev_recall_sarcasm": recall[1],
                "dev_f1_score_nonsarcasm":f1[0],
                "dev_precision_nonsarcasm": precision[0],
                "dev_recall_nonsarcasm":recall[0]
            }
            print (dev_result)
            print(sklearn.metrics.confusion_matrix(y_actual, predictions))

            if writer:
                writer.add_summary(summaries, step)
            return (accuracy,predictions,f1[1],dev_result)



        def test_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a test set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy ,predictions,y_actual= sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy , cnn.predictions,cnn.y],
                feed_dict)
            #time_str = datetime.datetime.now().isoformat()

            f1 = f1_score(y_actual, predictions, average=None)
            precision = precision_score(y_actual, predictions, average=None)
            recall = recall_score(y_actual, predictions, average=None)
            test_result = {
                "test_f1_score_sarcasm": f1[1],
                "test_precision_sarcasm": precision[1],
                "test_recall_sarcasm": recall[1],
                "test_f1_score_nonsarcasm":f1[0],
                "test_precision_nonsarcasm": precision[0],
                "test_recall_nonsarcasm":recall[0]
            }
            print (test_result)
            print(sklearn.metrics.confusion_matrix(y_actual, predictions))
            return test_result
            # if writer:
            #     writer.add_summary(summaries, step)


        if __name__ == "__main__":
            # Save the maximum accuracy value for validation data
            sess.run(tf.global_variables_initializer())

            max_result = {
                "max_f1_dev_sarcasm": "0.",
                "max_precision_dev_sarcasm": "0.",
                "max_recall_dev_sarcasm": "0.",
                "max_f1_test_sarcasm": "0.",
                "max_precision_test_sarcasm": "0.",
                "max_recall_test_sarcasm": "0.",
                "max_f1_dev_nonsarcasm": "0.",
                "max_precision_dev_nonsarcasm": "0.",
                "max_recall_dev_nonsarcasm": "0.",
                "max_f1_test_nonsarcasm": "0.",
                "max_precision_test_nonsarcasm": "0.",
                "max_recall_test_nonsarcasm": "0.",
                "max_epoch": "0"
            }

            converge_epoch = 0
            for epoch in range(FLAGS.num_epochs):
                time_start = time.time()
                epochs = data_helpers.epochs_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
                for batch in epochs:
                    x_batch , y_batch = zip(*batch)
                    train_accuracy = train_step(x_batch , y_batch)
                    if train_accuracy == 1 and converge_epoch == 0:
                        converge_epoch = epoch
                        print ("\nConverge_epoch:",epoch)
                    current_step = tf.train.global_step(sess, global_step)

                print("\nEvaluation:")
                print("Epoch: %03d" % (epoch))
                dev_accuracy, dev_predictions,dev_f1,dev_result = dev_step(x_dev, y_dev, writer=dev_summary_writer)

                if dev_f1 > float(max_result["max_f1_dev_sarcasm"]):

                    test_result = test_step(x_test, y_test, writer=dev_summary_writer)
                    #dev
                    max_result["max_f1_dev_sarcasm"] = dev_result["dev_f1_score_sarcasm"]
                    max_result["max_precision_dev_sarcasm"] = dev_result["dev_precision_sarcasm"]
                    max_result["max_recall_dev_sarcasm"] = dev_result["dev_recall_sarcasm"]
                    max_result["max_f1_dev_nonsarcasm"] = dev_result["dev_f1_score_nonsarcasm"]
                    max_result["max_precision_dev_nonsarcasm"] = dev_result["dev_precision_nonsarcasm"]
                    max_result["max_recall_dev_nonsarcasm"] = dev_result["dev_recall_nonsarcasm"]
                    #test
                    max_result["max_f1_test_sarcasm"] = test_result["test_f1_score_sarcasm"]
                    max_result["max_precision_test_sarcasm"] = test_result["test_precision_sarcasm"]
                    max_result["max_recall_test_sarcasm"] = test_result["test_recall_sarcasm"]
                    max_result["max_f1_test_nonsarcasm"] = test_result["test_f1_score_nonsarcasm"]
                    max_result["max_precision_test_nonsarcasm"] = test_result["test_precision_nonsarcasm"]
                    max_result["max_recall_test_nonsarcasm"] = test_result["test_recall_nonsarcasm"]
                    max_result["max_epoch"] = epoch
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("max_f1_dev %f" %max_result["max_f1_dev_sarcasm"])


                time_end = time.time()
                print ("time in one epoch",time_end-time_start)
                if (epoch - max_result["max_epoch"]) > 5:
                    break
            with open("result.csv", "a", encoding='utf8', newline='') as c:
                writer = csv.writer(c)
                if FLAGS.model == "cnn" or FLAGS.model == "twolayerCNN" or FLAGS.model == "highway_layer_cnn":
                    writer.writerow([FLAGS.model, FLAGS.num_filters, FLAGS.filter_sizes, FLAGS.dropout_keep_prob
                                     ,FLAGS.batch_size,FLAGS.learning_rate,FLAGS.l2_reg_lambda,
                                     max_result["max_f1_dev_sarcasm"] , max_result["max_precision_dev_sarcasm"] ,
                                     max_result["max_recall_dev_sarcasm"] , max_result["max_f1_dev_nonsarcasm"] ,
                                     max_result["max_precision_dev_nonsarcasm"] , max_result["max_recall_dev_sarcasm"],
                                     max_result["max_f1_test_sarcasm"] , max_result["max_precision_test_sarcasm"] ,
                                     max_result["max_recall_test_sarcasm"], max_result["max_f1_test_nonsarcasm"] ,
                                     max_result["max_precision_test_nonsarcasm"] , max_result["max_recall_test_sarcasm"],
                                     max_result["max_epoch"],converge_epoch
                                     ])
                else:
                    writer.writerow([FLAGS.model, FLAGS.hidden_sizes, FLAGS.num_layers, FLAGS.dropout_keep_prob,
                                     FLAGS.batch_size, FLAGS.learning_rate, FLAGS.l2_reg_lambda,
                                     max_result["max_f1_dev_sarcasm"] , max_result["max_precision_dev_sarcasm"] ,
                                     max_result["max_recall_dev_sarcasm"], max_result["max_f1_dev_nonsarcasm"],
                                     max_result["max_precision_dev_nonsarcasm"],max_result["max_recall_dev_sarcasm"],
                                     max_result["max_f1_test_sarcasm"],max_result["max_precision_test_sarcasm"],
                                     max_result["max_recall_test_sarcasm"],max_result["max_f1_test_nonsarcasm"],
                                     max_result["max_precision_test_nonsarcasm"],max_result["max_recall_test_sarcasm"],
                                     max_result["max_epoch"],converge_epoch
                                     ])