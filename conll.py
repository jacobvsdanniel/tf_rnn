import os
import sys
import time
import math
import random
import argparse
from collections import defaultdict

import numpy as np
import tensorflow as tf

import tf_rnn
import conll_utils

data_path = "../CONLL2012-intern/conll-2012/v4/data"
data_split_list = ["train", "development", "test"]

batch_nodes = 1000
batch_trees = 16
patience = 6
max_epoches = 60

def read_glove_embedding(model, word_to_index):
    # Read glove embeddings
    glove_word_array = np.load("glove_word.npy")
    glove_embedding_array = np.load("glove_embedding.npy")
    glove_word_to_index = {word: i for i, word in enumerate(glove_word_array)}
    
    # Initialize word embeddings to glove
    L = model.sess.run(model.L)
    for word, index in word_to_index.iteritems():
        if word in glove_word_to_index:
            L[index] = glove_embedding_array[glove_word_to_index[word]]
    model.sess.run(model.L.assign(L))
    return
    
def read_collobert_embedding(model, word_to_index):
    # Read glove embeddings
    word_array = np.load("collobert_word.npy")
    embedding_array = np.load("collobert_embedding.npy")
    collobert_to_index = {word: i for i, word in enumerate(word_array)}
    
    # Initialize word embeddings to glove
    L = model.sess.run(model.L)
    for word, index in word_to_index.iteritems():
        if word in collobert_to_index:
            L[index] = embedding_array[collobert_to_index[word]]
    model.sess.run(model.L.assign(L))
    return
    
def train():
    # Read data
    data, degree, word_to_index, labels, poses, characters, ne_list = (
        conll_utils.read_conll_dataset(raw_data_path=data_path))

    # Initialize model
    config = tf_rnn.Config()
    config.alphabet_size = characters
    config.pos_dimension = poses
    config.vocabulary_size = len(word_to_index)
    config.output_dimension = labels
    config.degree = degree
    model = tf_rnn.RNN(config)
    model.sess = tf.Session()
    model.sess.run(tf.initialize_all_variables())
    read_glove_embedding(model, word_to_index)
    # read_collobert_embedding(model, word_to_index)
    
    # Train
    saver = tf.train.Saver()
    best_score = (-1, -1, -1)
    best_epoch = 0
    for epoch in xrange(1, max_epoches+1):
        print "\n<Epoch %d>" % epoch
        
        start_time = time.time()
        loss = train_dataset(model, data["train"])
        print "[train] average loss %.3f; elapsed %.0fs" % (loss, time.time() - start_time)
        
        score = evaluate_dataset(model, data["development"], ne_list)
        print "[validation] precision=%.1f%% recall=%.1f%% f1=%.1f%%" % score
        
        if best_score[2] < score[2]:
            best_score = score
            best_epoch = epoch
            saver.save(model.sess, "tmp.model")
        elif epoch-best_epoch >= patience:
            break
    
    print "[best validation] precision=%.1f%% recall=%.1f%% f1=%.1f%%" % best_score
    saver.restore(model.sess, "tmp.model")
    score = evaluate_dataset(model, data["test"], ne_list)
    print "[test] precision=%.1f%% recall=%.1f%% f1=%.1f%%" % score
   
def make_tree_batch(tree_list):
    tree_list = sorted(tree_list, key=lambda tree: tree.nodes)
    
    batch_list = []
    batch = []
    nodes = 0
    for tree in tree_list:
        # if tree.nodes >= 100: continue
        if len(batch)>=batch_trees or nodes+tree.nodes>batch_nodes:
            batch_list.append(batch)
            batch = []
            nodes = 0
        batch.append(tree)
        nodes += tree.nodes
    batch_list.append(batch)
    
    # random.shuffle(batch_list)
    return batch_list

def make_tree_ner_batch(tree_list, ner_list):
    data = zip(tree_list, ner_list)
    data = sorted(data, key=lambda x: x[0].nodes)
    
    batch_list = []
    batch = []
    nodes = 0
    for tree, ner in data:
        # if tree.nodes >= 100: continue
        if len(batch)>=batch_trees or nodes+tree.nodes>batch_nodes:
            batch_list.append(batch)
            batch = []
            nodes = 0
        batch.append((tree, ner))
        nodes += tree.nodes
    batch_list.append(batch)
    
    return batch_list
    
def train_dataset(model, data):
    tree_list, _, _, _ = data
    batch_list = make_tree_batch(tree_list)
    print "batches:", len(batch_list)
    
    total_trees = sum(len(batch) for batch in batch_list)
    trees = 0
    total_loss = 0.
    for i, batch in enumerate(batch_list):
        loss = model.train(batch)
        total_loss += loss
        
        trees += len(batch)
        sys.stdout.write("\r(%5d/%5d) average loss %.3f" % (trees, total_trees, total_loss/trees))
    sys.stdout.write("\r" + " "*64 + "\r")
    return total_loss / total_trees

def evaluate_dataset(model, data, ne_list):
    tree_list, _, _, ner_list = data
    batch_list = make_tree_ner_batch(tree_list, ner_list)
    print "batches:", len(batch_list)
    
    total_true_postives = 0.
    total_postives = 0.
    total_reals = 0.
    for batch in batch_list:
        tree_list, ner_list = zip(*batch)
        true_postives, postives, reals = model.evaluate(tree_list, ner_list, ne_list)
        total_true_postives += true_postives
        total_postives += postives
        total_reals += reals
    print "true_postives", total_true_postives
    print "positives", total_postives
    print "reals", total_reals
    
    try:
        precision = total_true_postives / total_postives
    except ZeroDivisionError:
        precision = 1.
    recall = total_true_postives / total_reals
    f1 = 2*precision*recall / (precision + recall)
    return precision*100, recall*100, f1*100
    
def evaluate_confusion(model, data):
    tree_list, _, _, _ = data
    
    confusion_matrix = np.zeros([19, 19], dtype=np.int32)
    for tree in tree_list:
        confusion_matrix += model.predict(tree)
        
    return confusion_matrix

def validate(split):
    # Read data
    data, degree, word_to_index, labels, poses, characters, ne_list = (
        conll_utils.read_conll_dataset(raw_data_path=data_path))

    # Initialize model
    config = tf_rnn.Config()
    config.alphabet_size = characters
    config.pos_dimension = poses
    config.vocabulary_size = len(word_to_index)
    config.output_dimension = labels
    config.degree = degree
    model = tf_rnn.RNN(config)
    model.sess = tf.Session()
    model.sess.run(tf.initialize_all_variables())
    
    saver = tf.train.Saver()
    
    # TMP
    # for split in ["train", "development", "test"]:
        # saver.restore(model.sess, "tmp.model")
        # score = evaluate_dataset(model, data[split], ne_list)
        # print "[%s]" % split + " precision=%.1f%% recall=%.1f%% f1=%.1f%%" % score
    # for i, j in tmp_dict.iteritems():
        # print i, j
    # print len(tmp_dict)
    # return
    
    saver.restore(model.sess, "tmp.model")
    score = evaluate_dataset(model, data[split], ne_list)
    print "[%s]" % split + " precision=%.1f%% recall=%.1f%% f1=%.1f%%" % score
    return
    confusion_matrix = evaluate_confusion(model, data[split])
    
    ne_list.append("NONE")
    print " "*13,
    for ne in ne_list:
        print "%4s" % ne[:4],
    print ""
    for i in range(19):
        print "%12s" % ne_list[i],
        for j in range(19):
            if confusion_matrix[i][j]:
                print "%4d" % confusion_matrix[i][j],
            else:
                print "   .",
        print ""
    return

def interpolate_embedding():
    # Read data
    data, degree, word_to_index, labels, poses, characters, ne_list = (
        conll_utils.read_conll_dataset(raw_data_path=data_path))

    # Initialize model
    config = tf_rnn.Config()
    config.alphabet_size = characters
    config.pos_dimension = poses
    config.vocabulary_size = len(word_to_index)
    config.output_dimension = labels
    config.degree = degree
    model = tf_rnn.RNN(config)
    model.sess = tf.Session()
    model.sess.run(tf.initialize_all_variables())
    
    # Read glove embeddings
    glove_word_array = np.load("glove_word.npy")
    glove_embedding_array = np.load("glove_embedding.npy")
    glove_word_to_index = {word: i for i, word in enumerate(glove_word_array)}
    
    # Get glove embeddings
    L1 = model.sess.run(model.L)
    for word, index in word_to_index.iteritems():
        if word in glove_word_to_index:
            L1[index] = glove_embedding_array[glove_word_to_index[word]]
    
    # Get tuned embeddings
    saver = tf.train.Saver()
    saver.restore(model.sess, "tmp.model")
    L2 = model.sess.run(model.L)
    
    diff = np.any(L1!=L2, axis=1)
    print "Tuned words: %d" % np.sum(diff)
    same = np.all(L1==L2, axis=1)
    print "Un-tuned words: %d" % np.sum(same)
    
    print "Interpolating un-tuned embeddings..."
    start_time = time.time()
    L3 = np.copy(L2)
    for i in range(len(word_to_index)):
        if i%100==0 or i==len(word_to_index)-1:
            sys.stdout.write("%d\r" % i)
            sys.stdout.flush()
        if diff[i]: continue
        
        distance = np.linalg.norm(L1[i]-L1, axis=1)
        neighbor_index = np.argsort(distance)[1:11]
        
        distance = np.array([distance[j] for j in neighbor_index])
        distance_product = np.multiply.reduce(distance)
        normalizer = distance_product / np.sum(distance_product/distance)
        
        neighbor_embedding = L2[neighbor_index]
        L3[i] = normalizer * np.sum(neighbor_embedding / distance.reshape((10,1)), axis=0)
    print " elapsed %.0fs" % (time.time()-start_time)
    model.sess.run(model.L.assign(L3))
    saver.save(model.sess, "tmp2.model")
    return
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", dest="mode", default="train",
        choices=["train", "validate", "interpolate"])
    parser.add_argument("-s", dest="split", default="development",
        choices=["train", "development", "test"])
    arg = parser.parse_args()
    
    if arg.mode == "train":
        train()
    elif arg.mode == "validate":
        validate(arg.split)
    elif arg.mode == "interpolate":
        interpolate_embedding()
    return
    
if __name__ == "__main__":
    main()

    
    
