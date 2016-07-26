import os
import sys
import time
import random

import numpy as np
import tensorflow as tf

import tf_rnn
import conll_utils

data_path = "../CONLL2012-intern/conll-2012/v4/data"
data_split_list = ["train", "development", "test"]
glove_path = "."

patience = 3
max_epoches = 30

learning_rate = 0.001

embedding_dimension = 300
hidden_dimension = 300

def train():
    # Read data
    data, degree, word_to_index, labels, poses = (
        conll_utils.read_conll_dataset(raw_data_path=data_path))

    # Initialize model
    config = tf_rnn.Config(
        pos_dimension = poses,
        embedding_dimension = embedding_dimension,
        vocabulary_size = len(word_to_index), 
        hidden_dimension = hidden_dimension,
        output_dimension = labels,
        degree = degree,
        learning_rate = learning_rate)
    model = tf_rnn.RNN(config)
    model.sess = tf.Session()
    model.sess.run(tf.initialize_all_variables())
    
    # Initialize word embeddings to glove
    L = model.sess.run(model.L)
    glove_vecs = np.load(os.path.join(glove_path, "glove.npy"))
    glove_words = np.load(os.path.join(glove_path, "words.npy"))
    glove_word2idx = dict((word, i) for i, word in enumerate(glove_words))
    for word, index in word_to_index.iteritems():
        if word in glove_word2idx:
            L[index] = glove_vecs[glove_word2idx[word]]
    glove_vecs, glove_words, glove_word2idx = [], [], []
    update_op = model.L.assign(L)
    model.sess.run(update_op)

    # Train
    saver = tf.train.Saver()
    best_score = [0., 0., 0.]
    best_epoch = 0
    for epoch in xrange(max_epoches):
        if epoch - best_epoch > patience:      
            break
        print "\n<Epoch %d>" % epoch
        
        start_time = time.time()
        loss = train_dataset(model, data["train"])
        print "[train] average loss %.3f; elapsed %.2fs" % (loss, time.time() - start_time)
        
        score = evaluate_dataset(model, data["development"])
        print "[validation] precision=%.1f%% recall=%.1f%% f1=%.1f%%" % score
        
        if best_score[2] < score[2]:
            best_score = score
            best_epoch = epoch
            saver.save(model.sess, "tmp.model")
    
    print "[best validation] precision=%.1f%% recall=%.1f%% f1=%.1f%%" % best_score
    saver.restore(model.sess, "tmp.model")
    score = evaluate_dataset(model, data["test"])
    print "[test] precision=%.1f%% recall=%.1f%% f1=%.1f%%" % score

def train_dataset(model, data):
    tree_list, nodes, nes = data
    total_data = len(tree_list)
    total_loss = 0.
    #for i, tree in enumerate(tree_list):
    for i in range(total_data):
        tree = tree_list[random.randint(0,total_data-1)]
        loss = model.train(tree)
        total_loss += loss
        index = i + 1
        sys.stdout.write("\r(%5d/%5d) average loss %.3f" % (index, total_data, total_loss/index))
    print ""
    return total_loss / total_data

def evaluate_dataset(model, data):
    tree_list, nodes, nes = data
    
    total_true_postives = 0.
    total_postives = 0.
    for tree in tree_list:
        true_postives, postives = model.evaluate(tree)
        total_true_postives += true_postives
        total_postives += postives
    
    precision = total_true_postives / total_postives
    recall = total_true_postives / nes
    f1 = 2 / (1/precision + 1/recall)
    return precision*100, recall*100, f1*100

if __name__ == '__main__':
    train()
