import os
import sys
import time
import random
import codecs
import argparse

import numpy as np
import tensorflow as tf

import rnn

batch_nodes = 500
batch_trees = 16
patience = 20
max_epoches = 100

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
    
def load_lexicon_embedding(model, dataset):
    """ Load lexicon embeddings into the lexicon dictionary of model
    
    lexicon_embedding.npy: a 2d array of phrase vectors
    """
    embedding_array = np.load(os.path.join(dataset, "lexicon_embedding.npy"))
    model.sess.run(model.L_phrase.assign(embedding_array))
    return
    
def load_data_and_initialize_model(dataset, split_list=["train", "validate", "test"],
        use_pretrained_embedding=True):
    """ Get tree data and initialize a model
    
    #data: a dictionary; key-value example: "train"-(tree_list, ner_list)
    data: a dictionary; key-value example:
        "train"-{"tree_pyramid_list": tree_pyramid_list, "ner_list": ner_list}
    tree_pyramid_list: a list of (tree, pyramid) tuples
    ner_list: a list of dictionaries; key-value example: (3,5)-"PERSON"
    ne_list: a list of distinct string labels, e.g. "PERSON"
    """
    # Select the implementation of loading data according to dataset
    if dataset == "ontonotes":
        import ontonotes as data_utils
    elif dataset == "ontochinese":
        import ontochinese as data_utils
    elif dataset == "conll2003":
        import conll2003 as data_utils
    elif dataset == "conll2003dep":
        import conll2003dep as data_utils
    
    # Load data and determine dataset related hyperparameters
    config = rnn.Config()
    (data, word_list, ne_list,
        config.alphabet_size, config.pos_dimension, config.output_dimension, config.lexicons
        ) = data_utils.read_dataset(split_list)
    config.vocabulary_size = len(word_list)
    
    # Initialize a model
    model = rnn.RNN(config)
    """
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    model.sess = tf.Session(config=tf_config)
    """
    model.sess = tf.Session()
    model.sess.run(tf.global_variables_initializer())
    if use_pretrained_embedding: load_embedding(model, word_list, dataset)
    return data, ne_list, model

def make_batch_list(tree_pyramid_list):
    """ Create a list of batches of (tree, pyramid) tuples
    
    The (tree, pyramid) tuples in the same batch have similar numbers of nodes,
    so later padding can be minimized.
    """
    index_tree_pyramid_list = sorted(enumerate(tree_pyramid_list),
        key=lambda x: x[1][0].nodes+len(x[1][1]))
    
    batch_list = []
    batch = []
    for index, tree_pyramid in index_tree_pyramid_list:
        nodes = tree_pyramid[0].nodes + len(tree_pyramid[1])
        if len(batch)+1 > batch_trees or (len(batch)+1)*nodes > batch_nodes:
            batch_list.append(batch)
            batch = []
        batch.append((index, tree_pyramid))
    batch_list.append(batch)
    
    random.shuffle(batch_list)
    #batch_list = batch_list[::-1]
    return batch_list
    
def train_an_epoch(model, tree_pyramid_list):
    """ Update model parameters for every tree once
    """
    batch_list = make_batch_list(tree_pyramid_list)
    
    total_trees = len(tree_pyramid_list)
    trees = 0
    loss = 0.
    for i, batch in enumerate(batch_list):
        _, tree_pyramid_list = zip(*batch)
        #print "YOLO %d %d" % (tree_pyramid_list[-1][0].nodes, len(tree_pyramid_list[-1][1]))
        loss += model.train(tree_pyramid_list)
        trees += len(batch)
        sys.stdout.write("\r(%5d/%5d) average loss %.3f   " % (trees, total_trees, loss/trees))
        sys.stdout.flush()
    
    sys.stdout.write("\r" + " "*64 + "\r")
    return loss / total_trees

def predict_dataset(model, tree_pyramid_list, ne_list):
    """ Get dictionarues of predicted positive spans and their labels for every tree
    """
    batch_list = make_batch_list(tree_pyramid_list)
    
    ner_list = [None] * len(tree_pyramid_list)
    for batch in batch_list:
        index_list, tree_pyramid_list = zip(*batch)
        for i, span_y in enumerate(model.predict(tree_pyramid_list)):
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
    
def train_model(dataset, pretrain):
    """ Update model parameters until it converges or reaches maximum epochs
    """
    data, ne_list, model = load_data_and_initialize_model(dataset,
        use_pretrained_embedding=not pretrain)
    
    saver = tf.train.Saver()
    if pretrain:
        saver.restore(model.sess, "./tmp.model")
    
    best_epoch = 0
    best_score = (-1, -1, -1)
    best_loss = float("inf")
    for epoch in xrange(1, max_epoches+1):
        print "\n<Epoch %d>" % epoch
        
        start_time = time.time()
        loss = train_an_epoch(model, data["train"]["tree_pyramid_list"])
        print "[train] average loss %.3f; elapsed %.0fs" % (loss, time.time()-start_time)
        
        start_time = time.time()
        ner_hat_list = predict_dataset(model, data["validate"]["tree_pyramid_list"], ne_list)
        score = evaluate_prediction(data["validate"]["ner_list"], ner_hat_list)
        print "[validate] precision=%.1f%% recall=%.1f%% f1=%.3f%%; elapsed %.0fs;" % (score+(time.time()-start_time,)),
        
        if best_score[2] < score[2]:
            print "best"
            best_epoch = epoch
            best_score = score
            best_loss = loss
            saver.save(model.sess, "tmp.model")
        else:
            print "worse #%d" % (epoch-best_epoch)
        if epoch-best_epoch >= patience: break
    
    print "\n<Best Epoch %d>" % best_epoch
    print "[train] average loss %.3f" % best_loss
    print "[validate] precision=%.1f%% recall=%.1f%% f1=%.3f%%" % best_score
    saver.restore(model.sess, "./tmp.model")
    ner_hat_list = predict_dataset(model, data["test"]["tree_pyramid_list"], ne_list)
    score = evaluate_prediction(data["test"]["ner_list"], ner_hat_list)
    print "[test] precision=%.1f%% recall=%.1f%% f1=%.3f%%" % score
    return

def ner_diff(ner_a_list, ner_b_list):
    """
    Compute the differences of two ner predictions
    
    ner_list: a list of the ner prediction of each sentence
    ner: a dict of span-ne pairs
    """
    sentences = len(ner_a_list)
    print "%d sentences" % sentences
    print "a: %d nes" % sum(len(ner) for ner in ner_a_list)
    print "b: %d nes" % sum(len(ner) for ner in ner_b_list)
    
    ner_aa_list = []
    ner_bb_list = []
    ner_ab_list = []
    for i in xrange(sentences):
        ner_aa = {span: ne for span, ne in ner_a_list[i].iteritems()}
        ner_bb = {span: ne for span, ne in ner_b_list[i].iteritems()}
        ner_ab = {}
        for span, ne in ner_aa.items():
            if span in ner_bb and ner_aa[span] == ner_bb[span]:
                del ner_aa[span]
                del ner_bb[span]
                ner_ab[span] = ne
        ner_aa_list.append(ner_aa)
        ner_bb_list.append(ner_bb)
        ner_ab_list.append(ner_ab)
    
    return ner_aa_list, ner_bb_list, ner_ab_list
    
def write_ner(target_file, text_raw_data, ner_list):
    """
    Write the ner prediction of each sentence to file,
    indexing the senteces from 0.
    
    ner_list: a list of the ner prediction of each sentence
    ner: a dict of span-ne pairs
    """
    print ""
    print target_file
    sentences = len(text_raw_data)
    print "%d sentences" % sentences
    
    with codecs.open(target_file, "w", encoding="utf8") as f:
        for i in xrange(sentences):
            if len(ner_list[i]) == 0: continue
            f.write("\n%d\n" % i)
            f.write("%s\n" % " ".join(text_raw_data[i]))
            for span, ne in ner_list[i].iteritems():
                text_chunk = " ".join(text_raw_data[i][span[0]:span[1]])
                f.write("%d %d %s <%s>\n" % (span[0], span[1], ne, text_chunk))
    
    print "%d nes" % sum(len(ner) for ner in ner_list)
    return

def read_ner(source_file):
    """
    Read the ner prediction of each sentence from file,
    
    index_ner: a dict of setence index-ner pairs
    ner: a dict of span-ne pairs
    """
    with codecs.open(source_file, "r", encoding="utf8") as f:
        line_list = f.readlines()
        
    index_ner = {}
    sentence_index = -1
    line_index = -1
    while line_index+1 < len(line_list):
        line_index += 1
        line = line_list[line_index].strip().split()
        if not line: continue
        if len(line) == 1:
            sentence_index = int(line[0])
            index_ner[sentence_index] = {}
            line_index += 1
        else:
            l, r, ne = line[:3]
            index_ner[sentence_index][(int(l),int(r))] = ne
    return index_ner

def evaluate_model(dataset, split):
    """ Compute the scores of an existing model
    """
    data, ne_list, model = load_data_and_initialize_model(dataset, split_list=[split])
    
    saver = tf.train.Saver()
    saver.restore(model.sess, "./tmp.model")
    ner_hat_list = predict_dataset(model, data[split]["tree_pyramid_list"], ne_list)
    score = evaluate_prediction(data[split]["ner_list"], ner_hat_list)
    print "[%s]" % split + " precision=%.1f%% recall=%.1f%% f1=%.3f%%" % score
    """
    # YOLO
    text_raw_data = [tree.text_raw_data for tree, pyramid in data[split]["tree_pyramid_list"]]
    false_negative, false_positive, correct = ner_diff(data[split]["ner_list"], ner_hat_list)
    write_ner("bi_fn.txt", text_raw_data, false_negative)
    write_ner("bi_fp.txt", text_raw_data, false_positive)
    """
    return

def compare_model(dataset, split, bad_ner_file, good_ner_file, diff_file):
    data, ne_list, model = load_data_and_initialize_model(dataset, split_list=[split])
    text_raw_data = [tree.text_raw_data for tree, pyramid in data[split]["tree_pyramid_list"]]
    gold_ner_list = data[split]["ner_list"]
    
    bad_index_ner = read_ner(bad_ner_file)
    print "bad: %d nes", sum(len(ner) for index, ner in bad_index_ner.iteritems())
    good_index_ner = read_ner(good_ner_file)
    print "good: %d nes", sum(len(ner) for index, ner in good_index_ner.iteritems())
    
    index_span = {}
    for sentence_index in bad_index_ner:
        bad_ner = bad_index_ner[sentence_index]
        good_ner = good_index_ner[sentence_index] if sentence_index in good_index_ner else {}
        gold_ner = gold_ner_list[sentence_index]
        span_set = set()
        for span, ne in bad_ner.iteritems():
            if span not in good_ner:
                span_set.add(span)
        if span_set:
            index_span[sentence_index] = span_set
    
    with codecs.open(diff_file, "w", encoding="utf8") as f:
        for sentence_index, span_set in sorted(index_span.iteritems()):
            f.write("\n%d\n" % sentence_index)
            f.write("%s\n" % " ".join(text_raw_data[sentence_index]))
            for span in span_set:
                text_chunk = " ".join(text_raw_data[sentence_index][span[0]:span[1]])
                bad_ne = bad_index_ner[sentence_index][span]
                gold_ne = gold_ner_list[sentence_index][span] if span in gold_ner_list[sentence_index] else "NONE"
                f.write("%d %d %s->%s <%s>\n" % (span[0], span[1], bad_ne, gold_ne, text_chunk))
    return

def show_tree(dataset, split, tree_index):
    data, _, _ = load_data_and_initialize_model(dataset, split_list=[split])
    tree = data[split]["tree_pyramid_list"][tree_index][0]
    tree.show_tree()
    return
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", dest="mode", default="train",
        choices=["train", "evaluate", "compare", "showtree"])
    parser.add_argument("-s", dest="split", default="validate",
        choices=["train", "validate", "test"])
    parser.add_argument("-d", dest="dataset", default="ontonotes",
        choices=["ontonotes", "ontochinese", "conll2003", "conll2003dep"])
    parser.add_argument("-p", dest="pretrain", action="store_true")
    parser.add_argument("-i", dest="tree_index", default="24")
    arg = parser.parse_args()
    
    if arg.mode == "train":
        train_model(arg.dataset, arg.pretrain)
    elif arg.mode == "evaluate":
        evaluate_model(arg.dataset, arg.split)
    elif arg.mode == "compare":
        compare_model(arg.dataset, arg.split, "bot_fp.txt", "bi_fp.txt", "bot_to_bi.txt")
    elif arg.mode == "showtree":
        show_tree(arg.dataset, arg.split, int(arg.tree_index))
    return
    
if __name__ == "__main__":
    main()

    
    
    
    
    
