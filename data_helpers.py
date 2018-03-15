import numpy as np
import re
import itertools
from collections import Counter
import os


def clean_str(string):
    from nltk.tokenize import TweetTokenizer
    gettokens = TweetTokenizer()
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub("#", "", string)
    string = re.sub("#not ", "", string.lower())
    string = (' ').join(gettokens.tokenize(string))
    #print (string)
    return string.strip().lower()



def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r",encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    print ("len of pos"+positive_data_file, len(positive_examples))
    negative_examples = list(open(negative_data_file, "r",encoding='latin-1').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    print ("len of neg"+negative_data_file,len(negative_examples))
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def load_data_and_labels_combine(pos_train,neg_train,pos_dev,neg_dev,pos_test,neg_test):

    # Load data from files
    train_pos = list(open(pos_train, "r",encoding='utf-8').readlines())
    dev_pos = list(open(pos_dev, "r",encoding='utf-8').readlines())
    test_pos = list(open(pos_test, "r",encoding='utf-8').readlines())
    positive_examples = train_pos + dev_pos + test_pos
    positive_examples = [s.strip() for s in positive_examples]
    print ("len of pos", len(positive_examples))
    train_neg = list(open(neg_train, "r",encoding='utf-8').readlines())
    dev_neg = list(open(neg_dev, "r",encoding='utf-8').readlines())
    test_neg = list(open(neg_test, "r",encoding='utf-8').readlines())
    negative_examples = train_neg + dev_neg + test_neg
    negative_examples = [s.strip() for s in negative_examples]
    print ("len of neg",len(negative_examples))
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]




def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def epochs_iter(data, batch_size, num_epochs,shuffle=True):
    """
    Generates a epochs iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    # Shuffle the data at each epoch
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield shuffled_data[start_index:end_index]