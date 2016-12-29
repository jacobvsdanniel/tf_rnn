import re
import os
import sys
import codecs
from collections import defaultdict

import numpy as np
import mafan

sys.path.append("/home/danniel/Desktop/CONLL2012-intern")
from load_conll import load_data
from pstree import PSTree

from rnn import Node

dataset = "ontochinese"
character_file = os.path.join(dataset, "character.txt")
word_file = os.path.join(dataset, "word.txt")
pos_file = os.path.join(dataset, "pos.txt")
ne_file = os.path.join(dataset, "ne.txt")
pretrained_word_file = os.path.join(dataset, "word.npy")
pretrained_embedding_file = os.path.join(dataset, "embedding.npy")

data_path_prefix = "/home/danniel/Desktop/CONLL2012-intern/conll-2012/v4/data"
test_auto_data_path_prefix = "/home/danniel/Downloads/wu_conll_test/v9/data"
data_path_suffix = "data/chinese/annotations"
    
glove_file = "/home/danniel/Downloads/Glove_CNA_ASBC_300d.vec"

def log(msg):
    sys.stdout.write(msg)
    sys.stdout.flush()
    return

def read_list_file(file_path):
    log("Read %s..." % file_path)
    
    with codecs.open(file_path, "r", encoding="utf8") as f:
        line_list = f.read().splitlines()
    line_to_index = {line: index for index, line in enumerate(line_list)}
    
    log(" %d lines\n" % len(line_to_index))
    return line_list, line_to_index

def extract_vocabulary_and_alphabet():
    log("extract_vocabulary_and_alphabet()...")
    
    character_set = set()
    word_set = set()
    for split in ["train", "development", "test"]:
        full_path = os.path.join(data_path_prefix, split, data_path_suffix)
        config = {"file_suffix": "gold_conll", "dir_prefix": full_path}
        raw_data = load_data(config)
        for document in raw_data:
            for part in raw_data[document]:
                for index, sentence in enumerate(raw_data[document][part]["text"]):
                    for word in sentence:
                        for character in word:
                            character_set.add(character)
                        word_set.add(word)
                        
    with codecs.open(word_file, "w", encoding="utf8") as f:
        for word in sorted(word_set):
            f.write(word + '\n')
    
    with codecs.open(character_file, "w", encoding="utf8") as f:
        for character in sorted(character_set):
            f.write(character + '\n')
    
    log(" done\n")
    return

def extract_glove_embeddings():
    log("extract_glove_embeddings()...")
    
    _, word_to_index = read_list_file(word_file)
    word_list = []
    embedding_list = []
    with codecs.open(glove_file, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip().split()
            word = mafan.simplify(line[0])
            if word not in word_to_index: continue
            embedding = np.array([float(i) for i in line[1:]])
            word_list.append(word)
            embedding_list.append(embedding)
    
    np.save(pretrained_word_file, word_list)
    np.save(pretrained_embedding_file, embedding_list)
    
    log(" %d pre-trained words\n" % len(word_list))
    return

def construct_node(node, tree, ner_raw_data, head_raw_data, text_raw_data,
                    character_to_index, word_to_index, pos_to_index,
                    pos_count, ne_count, pos_ne_count):
    pos = tree.label
    word = tree.word
    span = tree.span
    head = tree.head if hasattr(tree, "head") else head_raw_data[(span, pos)][1]
    ne = ner_raw_data[span] if span in ner_raw_data else "NONE"
    
    # Process pos info
    node.pos = pos
    node.pos_index = pos_to_index[pos]
    pos_count[pos] += 1
    
    # Process word info
    node.word_split = [character_to_index[character] for character in word] if word else []
    node.word_index = word_to_index[word] if word else -1
    
    # Process head info
    node.head_split = [character_to_index[character] for character in head]
    node.head_index = word_to_index[head]
    
    # Process ne info
    node.ne = ne
    if not node.parent or node.parent.span!=span:
        ne_count[ne] += 1
    if ne != "NONE":
        pos_ne_count[pos] += 1
    
    # Process span info
    node.span = span
    
    # Binarize children
    if len(tree.subtrees) > 2:
        side_child_pos = tree.subtrees[-1].label
        side_child_span = tree.subtrees[-1].span
        side_child_head = head_raw_data[(side_child_span, side_child_pos)][1]
        if side_child_head != head:
            sub_subtrees = tree.subtrees[:-1]
        else:
            sub_subtrees = tree.subtrees[1:]
        new_span = (sub_subtrees[0].span[0], sub_subtrees[-1].span[1])
        new_tree = PSTree(label=pos, span=new_span, subtrees=sub_subtrees)
        new_tree.head = head
        if side_child_head != head:
            tree.subtrees = [new_tree, tree.subtrees[-1]]
        else:
            tree.subtrees = [tree.subtrees[0], new_tree]
         
    # Process children
    nodes = 1
    for subtree in tree.subtrees:
        child = Node()
        node.add_child(child)
        child_nodes = construct_node(child, subtree, ner_raw_data, head_raw_data, text_raw_data,
            character_to_index, word_to_index, pos_to_index,
            pos_count, ne_count, pos_ne_count)
        nodes += child_nodes
    return nodes

def get_tree_data(raw_data, character_to_index, word_to_index, pos_to_index):
    log("get_tree_data()...")
    """ Get tree structured data from CoNLL 2012
    
    Stores into Node data structure
    """
    tree_list = []
    ner_list = []
    word_count = 0
    pos_count = defaultdict(lambda: 0)
    ne_count = defaultdict(lambda: 0)
    pos_ne_count = defaultdict(lambda: 0)
    
    for document in raw_data["auto"]:
        for part in raw_data["auto"][document]:
            
            ner_raw_data = defaultdict(lambda: {})
            for k, v in raw_data["gold"][document][part]["ner"].iteritems():
                ner_raw_data[k[0]][(k[1], k[2])] = v
            
            for index, parse in enumerate(raw_data["auto"][document][part]["parses"]):
                text_raw_data = raw_data["auto"][document][part]["text"][index]
                word_count += len(text_raw_data)
                
                if parse.subtrees[0].label == "NOPARSE": continue
                head_raw_data = raw_data["auto"][document][part]["heads"][index]
                
                root_node = Node()
                nodes = construct_node(
                   root_node, parse, ner_raw_data[index], head_raw_data, text_raw_data,
                   character_to_index, word_to_index, pos_to_index,
                   pos_count, ne_count, pos_ne_count)
                root_node.nodes = nodes
                
                tree_list.append(root_node)
                ner_list.append(ner_raw_data[index])
                
    log(" %d sentences\n" % len(tree_list))
    return tree_list, ner_list, word_count, pos_count, ne_count, pos_ne_count

def label_tree_data(node, pos_to_index, ne_to_index):
    node.y = ne_to_index[node.ne]
    # node.y = ne_to_index[":".join(node.ner)]
        
    for child in node.child_list:
        label_tree_data(child, pos_to_index, ne_to_index)
    return
    
def read_dataset(data_split_list = ["train", "validate", "test"]):
    # Read all raw data
    annotation_method_list = ["gold", "auto"]
    raw_data = {}
    for split in data_split_list:
        raw_data[split] = {}
        for method in annotation_method_list:
            if split == "test" and method == "auto":
                full_path = os.path.join(test_auto_data_path_prefix, "test", data_path_suffix)
            else:
                if split == "validate":
                    data_path_root = "development"
                else:
                    data_path_root = split
                full_path = os.path.join(data_path_prefix, data_path_root, data_path_suffix)
            config = {"file_suffix": "%s_conll" % method, "dir_prefix": full_path}
            raw_data[split][method] = load_data(config)
    
    # Read lists of annotations
    character_list, character_to_index = read_list_file(character_file)
    word_list, word_to_index = read_list_file(word_file)
    pos_list, pos_to_index = read_list_file(pos_file)
    ne_list, ne_to_index = read_list_file(ne_file)
    
    # Build a tree structure for each sentence
    data = {}
    word_count = {}
    pos_count = {}
    ne_count = {}
    pos_ne_count = {}
    for split in data_split_list:
        (tree_list, ner_list,
            word_count[split], pos_count[split], ne_count[split], pos_ne_count[split]) = (
            get_tree_data(raw_data[split], character_to_index, word_to_index, pos_to_index))
        sentences = len(tree_list)
        nodes = sum(pos_count[split].itervalues())
        nes = sum(pos_ne_count[split].itervalues())
        data[split] = [tree_list, ner_list]
        log("<%s>\n  %d sentences; %d nodes; %d named entities\n"
            % (split, sentences, nodes, nes))
    
    # Show POS distribution
    total_pos_count = defaultdict(lambda: 0)
    for split in data_split_list:
        for pos in pos_count[split]:
            total_pos_count[pos] += pos_count[split][pos]
    nodes = sum(total_pos_count.itervalues())
    print "\nTotal %d nodes" % nodes
    print "-"*50 + "\n   POS   count  ratio\n" + "-"*50
    for pos, count in sorted(total_pos_count.iteritems(), key=lambda x: x[1], reverse=True):
        print "%6s %7d %5.1f%%" % (pos, count, count*100./nodes)
    
    # Show number of tokens and NEs in each split
    reals = 0
    split_nes_dict = {}
    for split in data_split_list:
        if split == "test": continue
        split_nes_dict[split] = sum(len(ner) for ner in data[split][1])
        reals += split_nes_dict[split]
    print "\nTotal %d named entities" % reals
    print "-"*50 + "\n       split   token     NE\n" + "-"*50
    for split in data_split_list:
        if split == "test": continue
        print "%12s %7d %6d" % (split, word_count[split], split_nes_dict[split])
    
    # Show NE distribution
    total_ne_count = defaultdict(lambda: 0)
    for split in data_split_list:
        if split == "test": continue
        for ne in ne_count[split]:
            if ne == "NONE": continue
            total_ne_count[ne] += ne_count[split][ne]
    nes = sum(total_ne_count.itervalues())
    print "\nTotal %d spanned named entities" % nes
    print "-"*50 + "\n          NE  count  ratio\n" + "-"*50
    for ne, count in sorted(total_ne_count.iteritems(), key=lambda x: x[1], reverse=True):
        print "%12s %6d %5.1f%%" % (ne, count, count*100./nes)
    
    # Show POS-NE distribution
    total_pos_ne_count = defaultdict(lambda: 0)
    for split in data_split_list:
        if split == "test": continue
        for pos in pos_ne_count[split]:
            total_pos_ne_count[pos] += pos_ne_count[split][pos]
    print "-"*50 + "\n   POS     NE   total  ratio\n" + "-"*50
    for pos, count in sorted(total_pos_ne_count.iteritems(), key=lambda x: x[1], reverse=True):
        total = total_pos_count[pos]
        print "%6s %6d %7d %5.1f%%" % (pos, count, total, count*100./total)
    
    # Compute the mapping to labels
    ne_to_index["NONE"] = len(ne_to_index)
    
    # Add label to nodes
    for split in data_split_list:
        for tree in data[split][0]:
            label_tree_data(tree, pos_to_index, ne_to_index)
    return (data, word_list, ne_list,
            len(character_to_index), len(pos_to_index), len(ne_to_index))

if __name__ == "__main__":
    #extract_vocabulary_and_alphabet()
    #extract_glove_embeddings()
    read_dataset()
    exit()
    
    
    
    
    
