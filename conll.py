import os
import sys
import time
import random
import argparse

import numpy as np
import tensorflow as tf

import tf_rnn
import conll_utils

data_path = "../CONLL2012-intern/conll-2012/v4/data"
data_split_list = ["train", "development", "test"]

patience = 5
max_epoches = 30

embedding_dimension = 300
hidden_dimension = 300

def train():
    # Read data
    data, degree, word_to_index, labels, poses, ne_list = (
        conll_utils.read_conll_dataset(raw_data_path=data_path))

    # Initialize model
    config = tf_rnn.Config()
    config.pos_dimension = poses
    config.embedding_dimension = embedding_dimension
    config.vocabulary_size = len(word_to_index)
    config.hidden_dimension = hidden_dimension
    config.output_dimension = labels
    config.degree = degree
    model = tf_rnn.RNN(config)
    model.sess = tf.Session()
    model.sess.run(tf.initialize_all_variables())
    
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

    # Train
    saver = tf.train.Saver()
    best_score = (-1, -1, -1)
    best_epoch = -1
    for epoch in xrange(max_epoches):
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

def train_dataset(model, data):
    tree_list, nodes, nes, ner_list = data
    total_data = len(tree_list)
    total_loss = 0.
    #for i, tree in enumerate(tree_list):
    for i in range(total_data):
        tree = tree_list[random.randint(0,total_data-1)]
        loss = model.train(tree)
        total_loss += loss
        index = i + 1
        sys.stdout.write("\r(%5d/%5d) average loss %.3f" % (index, total_data, total_loss/index))
    sys.stdout.write("\r" + " "*64 + "\r")
    return total_loss / total_data

def evaluate_dataset(model, data, ne_list):
    tree_list, nodes, nes, ner_list = data
    
    total_true_postives = 0.
    total_postives = 0.
    total_reals = 0.
    for index, tree in enumerate(tree_list):
        true_postives, postives, reals = model.evaluate(tree, ner_list[index], ne_list)
        total_true_postives += true_postives
        total_postives += postives
        total_reals += reals
    
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
    data, degree, word_to_index, labels, poses, ne_list = (
        conll_utils.read_conll_dataset(raw_data_path=data_path))

    # Initialize model
    config = tf_rnn.Config()
    config.pos_dimension = poses
    config.embedding_dimension = embedding_dimension
    config.vocabulary_size = len(word_to_index)
    config.hidden_dimension = hidden_dimension
    config.output_dimension = labels
    config.degree = degree
    model = tf_rnn.RNN(config)
    model.sess = tf.Session()
    model.sess.run(tf.initialize_all_variables())
    
    saver = tf.train.Saver()
    
    # TMP
    # for path in ["tmp.model", "tmp2.model"]:
        # print "\n<%s>" % path
        # for split in ["development", "test"]:
            # saver.restore(model.sess, path)
            # score = evaluate_dataset(model, data[split])
            # print "[%s]" % split + " precision=%.1f%% recall=%.1f%% f1=%.1f%%" % score
    # return
    
    saver.restore(model.sess, "tmp.model")
    score = evaluate_dataset(model, data[split], ne_list)
    print "[%s]" % split + " precision=%.1f%% recall=%.1f%% f1=%.1f%%" % score
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
    data, degree, word_to_index, labels, poses, ne_list = (
        conll_utils.read_conll_dataset(raw_data_path=data_path))

    # Initialize model
    config = tf_rnn.Config()
    config.pos_dimension = poses
    config.embedding_dimension = embedding_dimension
    config.vocabulary_size = len(word_to_index)
    config.hidden_dimension = hidden_dimension
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

    
    