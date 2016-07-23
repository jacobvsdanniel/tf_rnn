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
    
def get_sorted_dict(d):
    return sorted(zip(*reversed(zip(*d.iteritems()))), reverse=True)
    
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

def construct_node(tree, ner_raw_data, word_to_index, 
                    pos_count, ne_count, pos_ne_count,
                    sentence_index, offset):
    if len(tree.subtrees) == 1:
        return construct_node(tree.subtrees[0], ner_raw_data, word_to_index, 
                pos_count, ne_count, pos_ne_count,
                sentence_index, offset)
    start_offset = offset
    word = tree.word
    pos = tree.label
    
    if word:
        node = tree_rnn.Node(word_to_index[word])
        offset += 1
    else:
        node = tree_rnn.Node()
    
    degree = len(tree.subtrees)
    for subtree in tree.subtrees:
        child, child_degree, offset = construct_node(subtree, ner_raw_data, word_to_index, 
                                        pos_count, ne_count, pos_ne_count,
                                        sentence_index, offset)
        node.add_child(child)
        degree = max(degree, child_degree)
    
    span = (sentence_index, start_offset, offset)
    ne = ner_raw_data[span] if span in ner_raw_data else "NONE"
    node.ne = ne
    ne_count[ne] += 1
    
    node.pos = pos
    pos_count[pos] += 1
    if ne != "NONE": pos_ne_count[pos] += 1
    return node, degree, offset

def get_tree_data(raw_data, word_to_index):
    log("get_tree_data()...")
    """ Get tree structured data from CoNLL 2012
    
    Stores into tree_rnn.Node data structure
    """
    root_list = []
    max_degree = 0
    pos_count = defaultdict(lambda: 0)
    ne_count = defaultdict(lambda: 0)
    pos_ne_count = defaultdict(lambda: 0)
    
    for document in raw_data:
        for part in raw_data[document]:
            ner_raw_data = raw_data[document][part]["ner"]
            for sentence_index, parse in enumerate(raw_data[document][part]["parses"]):
                root_node, degree, _ = construct_node(
                                        parse, ner_raw_data, word_to_index,
                                        pos_count, ne_count, pos_ne_count,
                                        sentence_index, 0)
                root_list.append(root_node)
                max_degree = max(max_degree, degree)
            
    log(" %d sentences\n" % len(root_list))
    return root_list, max_degree, pos_count, ne_count, pos_ne_count

def label_tree_data(node, pos_ne_to_label):
    node.label = pos_ne_to_label[(node.pos, node.ne)]
        
    for child in node.children:
        if child:
            label_tree_data(child, pos_ne_to_label)
    return
    
def read_conll_dataset(raw_data_path = "../CONLL2012-intern/conll-2012/v4/data",
                       vocabulary_file = "vocab-cased.txt",
                       pos_file = "pos.txt",
                       named_entity_file = "ne.txt"):
    data_split_list = ["train", "development", "test"]
    
    # Read all raw data
    raw_data = {}
    for split in data_split_list:
        full_path = os.path.join(raw_data_path, split, "data/english/annotations")
        config = {"file_suffix": "gold_conll", "dir_prefix": full_path}
        raw_data[split] = load_data(config)
    
    # Extract word list
    vocabulary_set = set()
    for split in data_split_list:
        extract_vocabulary_from_raw_data(raw_data[split], vocabulary_set)
    word_to_index = get_word_to_index(vocabulary_set, vocabulary_file)
    
    # Read POS list
    pos_list, pos_to_index = read_pos(pos_file)
    
    # Read named entity list
    ne_list, ne_to_index = read_named_entity(named_entity_file)
    
    # Build a tree structure for each sentence
    data = {}
    max_degree = 0
    pos_count = {}
    ne_count = {}
    pos_ne_count = {}
    for split in data_split_list:
        tree_data, degree, pos_count[split], ne_count[split], pos_ne_count[split] = (
            get_tree_data(raw_data[split], word_to_index))
        sentences = len(tree_data)
        nodes = sum(pos_count[split].itervalues())
        nes = sum(pos_ne_count[split].itervalues())
        max_degree = max(max_degree, degree)
        data[split] = [tree_data, nodes, nes]
        log("<%s>\n  %d sentences; %d nodes; %d named entities\n"
            % (split, sentences, nodes, nes))
    log("degree %d\n" % max_degree)
    
    # Show POS distribution
    total_pos_count = defaultdict(lambda: 0)
    for split in data_split_list:
        for pos in pos_count[split]:
            total_pos_count[pos] += pos_count[split][pos]
    nodes = sum(total_pos_count.itervalues())
    print "\nTotal %d nodes" % nodes
    print "-"*50 + "\n   POS   count  ratio\n" + "-"*50
    for count, pos in get_sorted_dict(total_pos_count):
        print "%6s %7d %5.1f%%" % (pos, count, count*100./nodes)
    
    # Show NE distribution
    total_ne_count = defaultdict(lambda: 0)
    for split in data_split_list:
        for ne in ne_count[split]:
            if ne == "NONE": continue
            total_ne_count[ne] += ne_count[split][ne]
    nes = sum(total_ne_count.itervalues())
    print "\nTotal %d named entities" % nes
    print "-"*50 + "\n          NE  count  ratio\n" + "-"*50
    for count, ne in get_sorted_dict(total_ne_count):
        print "%12s %6d %5.1f%%" % (ne, count, count*100./nes)
    
    # Show POS-NE distribution
    total_pos_ne_count = defaultdict(lambda: 0)
    for split in data_split_list:
        if split == "test": continue
        for pos in pos_ne_count[split]:
            total_pos_ne_count[pos] += pos_ne_count[split][pos]
    print "-"*50 + "\n   POS     NE   total  ratio\n" + "-"*50
    for count, pos in get_sorted_dict(total_pos_ne_count):
        total = total_pos_count[pos]
        print "%6s %6d %7d %5.1f%%" % (pos, count, total, count*100./total)
    
    # Compute the mapping to label
    label = 0
    pos_ne_to_label = {}
    for ne in ne_list + ["NONE"]:
        for pos in pos_list:
            pos_ne_to_label[(pos, ne)] = label
            label += 1
    
    # Add label to nodes
    for split in data_split_list:
        for root_node in data[split][0]:
            label_tree_data(root_node, pos_ne_to_label)
    
    vocab = data_utils.Vocab()
    vocab.load(vocabulary_file)
    return data, max_degree, vocab, label
                
if __name__ == "__main__":
    read_conll_dataset()
    exit()
