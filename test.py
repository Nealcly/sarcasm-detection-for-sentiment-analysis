import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import os
import data_helpers
from tensorflow.contrib import learn
import time

# Parameters
# ==================================================

def sarcasm_detection(test_data):
    # change this to a directory with the desired checkpoint


    tf.flags.DEFINE_string("checkpoint_dir", ".\\runs\\1517968299\\checkpoints", "Checkpoint directory from training run")
    tf.flags.DEFINE_string("test_file", "twitter-datasets/test_data.txt", "Path and name of test file")
    tf.flags.DEFINE_string("submission_filename", "submission_predictions" + str(int(time.time())), "Path and name of submission file")

    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
    FLAGS = tf.flags.FLAGS



    #test_data = ["why do we always hurt the ones we love?"]
    test_data = [test_data]
    test_data = [data_helpers.clean_str(test) for test in test_data]
    print (test_data)

    # Map data into vocabulary
    #vocab_path = ".\\runs\\1516091567\\vocab"
    vocab_path = FLAGS.checkpoint_dir + "\\..\\vocab"
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(test_data)))


    # Evaluation
    # ==================================================
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            #input_x = graph.get_operation_by_name("train_input").outputs[0]
            #train_input
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            #predictions = graph.get_operation_by_name("Model/predictions").outputs[0]

            #Prediction
            predictions = sess.run(predictions, {input_x: x_test, dropout_keep_prob: 1.0})
            print (predictions)

            if predictions[0] == 0:
                return "non_sarcasm"
            else:
                return "sarcasm"



if __name__ == "__main__":

    test= "I just like bus"
    print (sarcasm_detection(test))
