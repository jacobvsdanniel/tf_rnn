import os
import sys
import time
import random
import argparse

import numpy as np
import tensorflow as tf

import rnn

batch_nodes = 1000
batch_trees = 16
patience = 3
max_epoches = 30

def load_embedding(model, word_list, dataset):
    """ Load pre-trained word embeddings into the dictionary of model
    
    word.npy: an array of pre-trained words
    embedding.npy: a 2d array of pre-trained word vectors
    word_list: a list of words in the dictionary of model
    """
    # load pre-trained word embeddings from file
    word_array = np.load(os.path.join(dataset, "word.npy"))
    embedding_array = np.load(os.path.join(dataset, "embedding.npy"))
    word_to_embedding = {}
    for i, word in enumerate(word_array):
        word_to_embedding[word] = embedding_array[i]
    
    # Store pre-trained word embeddings into the dictionary of model
    L = model.sess.run(model.L)
    for index, word in enumerate(word_list):
        if word in word_to_embedding:
            L[index] = word_to_embedding[word]
    model.sess.run(model.L.assign(L))
    return
    
def load_data_and_initialize_model(dataset, split_list=["train", "validate", "test"],
        use_pretrained_embedding=True):
    """ Get tree data and initialize a model
    
    data: a dictionary; key-value example: "train"-(tree_list, ner_list)
    ner_list: a list of dictionaries; key-value example: (3,5)-"PERSON"
    ne_list: a list of distinct string labels, e.g. "PERSON"
    """
    # Select the implementation of loading data according to dataset
    if dataset == "ontonotes":
        import ontonotes as data_utils
    elif dataset == "ontochinese":
        import ontochinese as data_utils
    
    # Load data and determine dataset related hyperparameters
    config = rnn.Config()
    (data, word_list, ne_list,
        config.alphabet_size, config.pos_dimension, config.output_dimension
        ) = data_utils.read_dataset(split_list)
    config.vocabulary_size = len(word_list)
    
    # Initialize a model
    model = rnn.RNN(config)
    model.sess = tf.Session()
    model.sess.run(tf.initialize_all_variables())
    if use_pretrained_embedding: load_embedding(model, word_list, dataset)
    return data, ne_list, model

def make_batch_list(tree_list):
    """ Create a list of batches of trees
    
    The trees in the same batch have similar numbers of nodes, so later padding can be minimized.
    """
    index_tree_list = sorted(enumerate(tree_list), key=lambda x: x[1].nodes)
    
    batch_list = []
    batch = []
    for index, tree in index_tree_list:
        if len(batch)+1>batch_trees or (len(batch)+1)*tree.nodes>batch_nodes:
            batch_list.append(batch)
            batch = []
        batch.append((index, tree))
    batch_list.append(batch)
    
    random.shuffle(batch_list)
    return batch_list
    
def train_an_epoch(model, tree_list):
    """ Update model parameters for every tree once
    """
    batch_list = make_batch_list(tree_list)
    
    total_trees = len(tree_list)
    trees = 0
    loss = 0.
    for i, batch in enumerate(batch_list):
        _, tree_list = zip(*batch)
        loss += model.train(tree_list)
        trees += len(batch)
        sys.stdout.write("\r(%5d/%5d) average loss %.3f   " % (trees, total_trees, loss/trees))
        sys.stdout.flush()
    
    sys.stdout.write("\r" + " "*64 + "\r")
    return loss / total_trees

def predict_dataset(model, tree_list, ne_list):
    """ Get dictionarues of predicted positive spans and their labels for every tree
    """
    batch_list = make_batch_list(tree_list)
    
    ner_list = [None] * len(tree_list)
    for batch in batch_list:
        index_list, tree_list = zip(*batch)
        for i, span_y in enumerate(model.predict(tree_list)):
            ner_list[index_list[i]] = {span: ne_list[y] for span, y in span_y.iteritems()}
    return ner_list
    
def evaluate_prediction(ner_list, ner_hat_list):
    """ Compute the score of the prediction of trees
    """
    reals = 0.
    positives = 0.
    true_positives = 0.
    for index, ner in enumerate(ner_list):
        ner_hat = ner_hat_list[index]
        reals += len(ner)
        positives += len(ner_hat)
        for span in ner_hat.iterkeys():
            if span not in ner: continue
            if ner[span] == ner_hat[span]:
                true_positives += 1
    
    try:
        precision = true_positives / positives
    except ZeroDivisionError:
        precision = 1.
    
    try:
        recall = true_positives / reals
    except ZeroDivisionError:
        recall = 1.
    
    try:
        f1 = 2*precision*recall / (precision + recall)
    except ZeroDivisionError:
        f1 = 0.
    
    return precision*100, recall*100, f1*100
    
def train_model(dataset):
    """ Update model parameters until it converges or reaches maximum epochs
    """
    if os.path.isfile("keep_training.model"):
        load_existing_model = True
    else:
        load_existing_model = False
        
    data, ne_list, model = load_data_and_initialize_model(dataset,
        use_pretrained_embedding=not load_existing_model)
    
    saver = tf.train.Saver()
    if load_existing_model:
        saver.restore(model.sess, "keep_training.model")
    
    best_epoch = 0
    best_score = (-1, -1, -1)
    best_loss = float("inf")
    for epoch in xrange(1, max_epoches+1):
        print "\n<Epoch %d>" % epoch
        
        start_time = time.time()
        loss = train_an_epoch(model, data["train"][0])
        print "[train] average loss %.3f; elapsed %.0fs" % (loss, time.time() - start_time)
        
        ner_hat_list = predict_dataset(model, data["validate"][0], ne_list)
        score = evaluate_prediction(data["validate"][1], ner_hat_list)
        print "[validate] precision=%.1f%% recall=%.1f%% f1=%.3f%%" % score,
        
        if best_score[2] < score[2]:
            print "best"
            best_epoch = epoch
            best_score = score
            best_loss = loss
            saver.save(model.sess, "tmp.model")
        else: print ""
        if epoch-best_epoch >= patience: break
    
    print "\n<Best Epoch %d>" % best_epoch
    print "[train] average loss %.3f" % best_loss
    print "[validate] precision=%.1f%% recall=%.1f%% f1=%.3f%%" % best_score
    saver.restore(model.sess, "tmp.model")
    ner_hat_list = predict_dataset(model, data["test"][0], ne_list)
    score = evaluate_prediction(data["test"][1], ner_hat_list)
    print "[test] precision=%.1f%% recall=%.1f%% f1=%.3f%%" % score
    return

def evaluate_model(dataset, split):
    """ Compute the scores of an existing model
    """
    data, ne_list, model = load_data_and_initialize_model(dataset, split_list=[split])
    
    saver = tf.train.Saver()
    saver.restore(model.sess, "tmp.model")
    ner_hat_list = predict_dataset(model, data[split][0], ne_list)
    score = evaluate_prediction(data[split][1], ner_hat_list)
    print "[%s]" % split + " precision=%.1f%% recall=%.1f%% f1=%.3f%%" % score
    return
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", dest="mode", default="train",
        choices=["train", "evaluate"])
    parser.add_argument("-s", dest="split", default="validate",
        choices=["train", "validate", "test"])
    parser.add_argument("-d", dest="dataset", default="ontonotes",
        choices=["ontonotes", "ontochinese"])
    arg = parser.parse_args()
    
    if arg.mode == "train":
        train_model(arg.dataset)
    elif arg.mode == "evaluate":
        evaluate_model(arg.dataset, arg.split)
    return
    
if __name__ == "__main__":
    main()

    
    
    
    
    
