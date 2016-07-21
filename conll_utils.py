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
    
def read_named_entity(named_entity_file):
    log("read_named_entity()...")
    entity_list = []
    entity_to_index = {}
    with open(named_entity_file, "r") as f:
        for line in f.readlines():
            entity = line.strip()
            entity_to_index[entity] = len(entity_list)
            entity_list.append(entity)
    log(" done\n")
    return entity_list, entity_to_index

def construct_node(tree, word_to_index, pos_to_index, entity_to_index, 
                   label_count, sentence, offset, ner_dict):
    if len(tree.subtrees) == 1:
        return construct_node(tree.subtrees[0], word_to_index, pos_to_index, entity_to_index,
                   label_count, sentence, offset, ner_dict)
    
    start_offset = offset
    
    if tree.word:
        node = tree_rnn.Node(word_to_index[tree.word])
        offset += 1
    else:
        node = tree_rnn.Node()
    
    # POS label
    # node.label = pos_to_index[tree.label]
    # label_count[tree.label] += 1
    
    degree = len(tree.subtrees)
    for subtree in tree.subtrees:
        child, child_degree, offset = construct_node(subtree,
                                          word_to_index, pos_to_index, entity_to_index,
                                          label_count, sentence, offset, ner_dict)
        node.add_child(child)
        degree = max(degree, child_degree)
    
    # Named entity label
    span = (sentence, start_offset, offset)
    if span in ner_dict:
        node.label = entity_to_index[ner_dict[span]]
        label_count[ner_dict[span]] += 1
    else:
        node.label = len(entity_to_index)
        label_count["NONE"] += 1
        
    return node, degree, offset

def get_tree_data(raw_data, word_to_index, pos_to_index, entity_to_index, label_count):
    log("get_tree_data()...")
    """ Get tree structured data from CoNLL 2012
    
    Stores into tree_rnn.Node data structure
    """
    root_list = []
    max_degree = 0
    
    for document in raw_data:
        for part in raw_data[document]:
            ner_dict = raw_data[document][part]["ner"]
            for i, parse in enumerate(raw_data[document][part]["parses"]):
                root_node, degree, _ = construct_node(
                                                parse,
                                                word_to_index, pos_to_index, entity_to_index,
                                                label_count, i, 0, ner_dict)
                root_list.append(root_node)
                max_degree = max(max_degree, degree)
            
    log(" %d sentences\n" % len(root_list))
    return root_list, max_degree

def read_conll_dataset(raw_data_path = "../CONLL2012-intern/conll-2012/v4/data",
                       vocabulary_file = "vocab-cased.txt",
                       pos_file = "pos.txt",
                       named_entity_file = "named_entity.txt"):
    data_split_list = ["train", "development", "test"]
    
    # Read all raw data
    raw_data = {}
    for data_split in data_split_list:
        full_path = os.path.join(raw_data_path, data_split, "data/english/annotations")
        config = {"file_suffix": "gold_conll", "dir_prefix": full_path}
        raw_data[data_split] = load_data(config)
    
    # Extract word list
    vocabulary_set = set()
    for data_split in data_split_list:
        extract_vocabulary_from_raw_data(raw_data[data_split], vocabulary_set)
    word_to_index = get_word_to_index(vocabulary_set, vocabulary_file)
    
    # Read POS list
    pos_list, pos_to_index = read_pos(pos_file)
    
    # Read named entity list
    entity_list, entity_to_index = read_named_entity(named_entity_file)
    
    # Build a tree structure for each sentence
    tree_data = {}
    max_degree = 0
    label_count = defaultdict(lambda: 0)
    for data_split in data_split_list:
        tree_data[data_split], degree = get_tree_data(
                                                raw_data[data_split],
                                                word_to_index, pos_to_index, entity_to_index,
                                                label_count)
        log("%s %d\n" % (data_split, len(tree_data[data_split])))
        max_degree = max(max_degree, degree)
    log("degree %d\n" % max_degree)
    
    # labels = len(pos_to_index)
    labels = len(entity_to_index)
    non_entities = label_count.pop("NONE")
    entities = sum(label_count.itervalues())
    log("labels %d; named entities %d; none entities %d\n" % (labels, entities, non_entities))
    for count, label in sorted(zip(*reversed(zip(*label_count.iteritems()))), reverse=True)[:18]:
        log("  %s %.1f%%\n" % (label, count*100./entities))
        
    vocab = data_utils.Vocab()
    vocab.load(vocabulary_file)
    return tree_data, max_degree, vocab, labels+1, entities, non_entities
                
if __name__ == "__main__":
    read_conll_dataset()
    exit()