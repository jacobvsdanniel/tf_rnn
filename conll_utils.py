import os
import sys
import codecs
from collections import defaultdict

sys.path.append("../CONLL2012-intern")
from load_conll import load_data

import tree_rnn
import data_utils

def log(msg):
    sys.stdout.write(msg)
    sys.stdout.flush()
    return

def extract_vocabulary_from_raw_data(raw_data, vocabulary_set):
    log("extract_vocabulary_from_raw_data()...")
    for document in raw_data:
        for part in raw_data[document]:
            for sentence in raw_data[document][part]["text"]:
                vocabulary_set |= set(sentence)
    log(" done\n")
    return
    
def get_word_to_index(vocabulary_set, vocabulary_file):
    log("get_word_to_index()...")
    word_to_index = {}
    with codecs.open(vocabulary_file, "w", encoding="utf8") as f:
        count = 0
        for word in sorted(vocabulary_set):
            word_to_index[word] = count
            count += 1
            f.write(word + '\n')
    log(" %d words\n" % count)
    return word_to_index

def read_pos(pos_file):
    log("read_pos()...")
    pos_list = []
    pos_to_index = {}
    with open(pos_file, "r") as f:
        for line in f.readlines():
            pos = line.strip()
            pos_to_index[pos] = len(pos_list)
            pos_list.append(pos)
    log(" done\n")
    return pos_list, pos_to_index

def construct_node(tree, word_to_index, pos_to_index, label_count):
    if tree.word:
        node = tree_rnn.Node(word_to_index[tree.word])
    else:
        node = tree_rnn.Node()
    
    node.label = pos_to_index[tree.label]
    label_count[tree.label] += 1
    
    degree = len(tree.subtrees)
    for subtree in tree.subtrees:
        child, child_degree = construct_node(subtree, word_to_index, pos_to_index, label_count)
        node.add_child(child)
        degree = max(degree, child_degree)
        #if child_degree == 63: print subtree
        
    return node, degree

def get_tree_data(raw_data, word_to_index, pos_to_index, label_count):
    log("get_tree_data()...")
    """ Get tree structured data from CoNLL 2012
    
    Stores into tree_rnn.Node data structure
    """
    root_list = []
    max_degree = 0
    
    for document in raw_data:
        for part in raw_data[document]:
            for parse in raw_data[document][part]["parses"]:
                root_node, degree = construct_node(parse, word_to_index, pos_to_index, label_count)
                root_list.append(root_node)
                max_degree = max(max_degree, degree)
    
    log(" %d sentences\n" % len(root_list))
    return root_list, max_degree

def read_conll_dataset(raw_data_path = "../CONLL2012-intern/conll-2012/v4/data",
                       vocabulary_file = "vocab-cased.txt",
                       pos_file = "pos.txt"):
    data_split_list = ["train", "development", "test"]
    
    raw_data = {}
    for data_split in data_split_list:
        full_path = os.path.join(raw_data_path, data_split, "data/english/annotations")
        config = {"file_suffix": "gold_conll", "dir_prefix": full_path}
        raw_data[data_split] = load_data(config)
    
    vocabulary_set = set()
    for data_split in data_split_list:
        extract_vocabulary_from_raw_data(raw_data[data_split], vocabulary_set)
    
    word_to_index = get_word_to_index(vocabulary_set, vocabulary_file)
    
    pos_list, pos_to_index = read_pos(pos_file)
    
    tree_data = {}
    max_degree = 0
    label_count = defaultdict(lambda: 0)
    for data_split in data_split_list:
        tree_data[data_split], degree = get_tree_data(
                                                raw_data[data_split],
                                                word_to_index,
                                                pos_to_index,
                                                label_count)
        log("%s %d\n" % (data_split, len(tree_data[data_split])))
        max_degree = max(max_degree, degree)
    log("degree %d\n" % max_degree)
    
    labels = len(label_count)
    nodes = sum(label_count.itervalues())
    log("labels %d; nodes %d\n" % (labels, nodes))
    for count, label in sorted(zip(*reversed(zip(*label_count.iteritems()))), reverse=True)[:10]:
        log("  %s %.1f%%\n" % (label, count*100./nodes))
        
    vocab = data_utils.Vocab()
    vocab.load(vocabulary_file)
    return tree_data, max_degree, vocab, len(pos_to_index)
                
if __name__ == "__main__":
    read_conll_dataset()
    exit()