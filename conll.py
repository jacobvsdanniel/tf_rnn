import os
import time

import numpy as np
import tensorflow as tf

import tf_rnn
import conll_utils

data_path = "../CONLL2012-intern/conll-2012/v4/data"
data_split_list = ["train", "development", "test"]
glove_path = "../treelstm/data"

max_epoches = 30
learning_rate = 0.001

embedding_dimension = 300
hidden_dimension = 100

def train():
    # Read data
    data, degree, vocab, labels = conll_utils.read_conll_dataset(raw_data_path=data_path)

    # Initialize model
    config = tf_rnn.Config(
        embedding_dimension = embedding_dimension,
        vocabulary_size = vocab.size(), 
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
    for i, word in enumerate(vocab.words):
        if word in glove_word2idx:
            L[i] = glove_vecs[glove_word2idx[word]]
    glove_vecs, glove_words, glove_word2idx = [], [], []
    update_op = model.L.assign(L)
    model.sess.run(update_op)

    # Train
    start_time = time.time()
    saver = tf.train.Saver()
    best_score = 0.
    best_epoch = 0
    for epoch in xrange(max_epoches):
        if epoch - best_epoch > model.config.patience:      
            break
        print "Epoch %d" % epoch
        
        loss = train_dataset(model, data["train"])
        print "\ntrain average loss %.1f" % loss
        
        score = evaluate_dataset(model, data["development"])
        print "validation accuracy %.1f%%" % (100*score)
        
        if best_score < score:
            best_score = score
            best_epoch = epoch
            saver.save(model.sess, "tmp.model")
        
    print "finished training, elapsed %.2fs" % (time.time() - start_time)
    saver.restore(model.sess, "tmp.model")
    score = evaluate_dataset(model, data["test"])
    print "test accuracy %.1f%%" % (100*score)

def train_dataset(model, data):
    total_data = len(data)
    total_loss = 0.
    for i, tree in enumerate(data):
        loss = model.train(tree)
        total_loss += loss
        index = i + 1
        print "average loss %.2f at example %d of %d\r" % (total_loss/index, index, total_data),
    return total_loss / total_data

def evaluate_dataset(model, data):
    total_corrects = 0.
    total_samples = 0
    for tree in data:
        corrects, samples = model.evaluate(tree)
        total_corrects += corrects
        total_samples += samples
    return total_corrects / total_samples

if __name__ == '__main__':
    train()
