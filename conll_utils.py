import os
import sys
import codecs
from collections import defaultdict

import numpy as np

sys.path.append("../CONLL2012-intern")
from load_conll import load_data
from pstree import PSTree

class Node(object):
    def __init__(self):
        self.parent = None
        self.child_list = []
        
    def add_child(self, child):
        self.child_list.append(child)
        child.parent = self

def log(msg):
    sys.stdout.write(msg)
    sys.stdout.flush()
    return
    
def get_sorted_dict(d):
    return sorted(zip(*reversed(zip(*d.iteritems()))), reverse=True)

def extract_glove_embeddings(glove_file = "glove.840B.300d.txt",
                             vocabulary_file = "vocabulary.txt"):
    
    word_list, word_to_index = read_vocabulary(vocabulary_file)
    
    word_list = []
    embedding_list = []
    with open(glove_file, "r") as f:
        for line in f:
            line = line.strip().split()
            word = line[0]
            if word not in word_to_index: continue
            embedding = np.array([float(i) for i in line[1:]])
            word_list.append(word)
            embedding_list.append(embedding)
    
    np.save("glove_word.npy", word_list)
    np.save("glove_embedding.npy", embedding_list)
    return
    
def extract_conll_vocabulary(raw_data_path = "../CONLL2012-intern/conll-2012/v4/data",
                             vocabulary_file = "vocabulary.txt"):
    log("extract_conll_vocabulary()...")
    
    vocabulary_set = set()
    for split in ["train", "development", "test"]:
        full_path = os.path.join(raw_data_path, split, "data/english/annotations")
        config = {"file_suffix": "gold_conll", "dir_prefix": full_path}
        raw_data = load_data(config)
        for document in raw_data:
            for part in raw_data[document]:
                for sentence in raw_data[document][part]["text"]:
                    vocabulary_set |= set(sentence)
    
    with codecs.open(vocabulary_file, "w", encoding="utf8") as f:
        for word in sorted(vocabulary_set):
            f.write(word + '\n')
        
    log(" done\n")
    return
    
def read_vocabulary(vocabulary_file):
    log("read_vocabulary()...")
    
    word_list = []
    word_to_index = {}
    with codecs.open(vocabulary_file, "r", encoding="utf8") as f:
        for line in f.readlines():
            word = line.strip()
            word_to_index[word] = len(word_list)
            word_list.append(word)
    
    log(" %d words\n" % len(word_to_index))
    return word_list, word_to_index

def read_pos(pos_file):
    log("read_pos()...")
    
    pos_list = []
    pos_to_index = {}
    with open(pos_file, "r") as f:
        for line in f.readlines():
            pos = line.strip()
            pos_to_index[pos] = len(pos_list)
            pos_list.append(pos)
    
    log(" %d pos\n" % len(pos_to_index))
    return pos_list, pos_to_index
    
def read_ne(ne_file):
    log("read_ne()...")
    
    ne_list = []
    ne_to_index = {}
    with open(ne_file, "r") as f:
        for line in f.readlines():
            ne = line.strip()
            ne_to_index[ne] = len(ne_list)
            ne_list.append(ne)
    
    log(" %d ne\n" % len(ne_to_index))
    return ne_list, ne_to_index

def construct_node(tree, ner_raw_data, head_raw_data, text_raw_data,
                    word_to_index, pos_to_index,
                    pos_count, ne_count, pos_ne_count,
                    sentence_index):
    if len(tree.subtrees) == 1:
        return construct_node(tree.subtrees[0], ner_raw_data, head_raw_data, text_raw_data,
                                word_to_index, pos_to_index,
                                pos_count, ne_count, pos_ne_count,
                                sentence_index)
    node = Node()
    pos = tree.label
    word = tree.word
    span = tree.span
    if hasattr(tree, "head"): 
        head = tree.head
    else:
        head = head_raw_data[(span, pos)][1]
    sentence_span = (sentence_index, span[0], span[1])
    ne = ner_raw_data[sentence_span] if sentence_span in ner_raw_data else "NONE"
    
    # Process pos info
    node.pos = pos
    node.pos_index = pos_to_index[pos]
    pos_count[pos] += 1
    # if not word: node.pos_index = -1
    
    # Process word info
    if word:
        node.word_index = word_to_index[word]
    else:
        node.word_index = -1
    
    window = 0
    node.window_index_list = [-1] * (2*window)
    for i in range(window):
        index = span[0]-1-i
        if index < 0: break
        node.window_index_list[window-1-i] = word_to_index[text_raw_data[index]]
    for i in range(window):
        index = span[1]+i
        if index >= len(text_raw_data): break
        node.window_index_list[window+i] = word_to_index[text_raw_data[index]]
    
    # Process head info
    node.head_index = word_to_index[head]
    node.parent_head_index = -1
    
    # Process ne info
    node.ne = ne
    ne_count[ne] += 1
    if ne != "NONE": pos_ne_count[pos] += 1
    
    # Binarize children
    degree = len(tree.subtrees)
    if degree > 2:
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
    degree = len(tree.subtrees)
    for subtree in tree.subtrees:
        child, child_degree = construct_node(subtree, ner_raw_data, head_raw_data, text_raw_data,
                                                word_to_index, pos_to_index,
                                                pos_count, ne_count, pos_ne_count,
                                                sentence_index)
        node.add_child(child)
        child.parent_head_index = node.head_index
        degree = max(degree, child_degree)
    
    return node, degree

def get_tree_data(raw_data, word_to_index, pos_to_index):
    log("get_tree_data()...")
    """ Get tree structured data from CoNLL 2012
    
    Stores into Node data structure
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
                head_raw_data = raw_data[document][part]["heads"][sentence_index]
                text_raw_data = raw_data[document][part]["text"][sentence_index]
                root_node, degree = construct_node(
                                        parse, ner_raw_data, head_raw_data, text_raw_data,
                                        word_to_index, pos_to_index,
                                        pos_count, ne_count, pos_ne_count,
                                        sentence_index)
                                        
                root_node.text = raw_data[document][part]["text"][sentence_index]
                                        
                root_list.append(root_node)
                max_degree = max(max_degree, degree)
            
    log(" %d sentences\n" % len(root_list))
    return root_list, max_degree, pos_count, ne_count, pos_ne_count

def label_tree_data(node, pos_to_index, ne_to_index):
    node.y1 = ne_to_index[node.ne]
    node.y2 = pos_to_index[node.pos]
        
    for child in node.child_list:
        label_tree_data(child, pos_to_index, ne_to_index)
    return
    
def read_conll_dataset(raw_data_path = "../CONLL2012-intern/conll-2012/v4/data",
                       vocabulary_file = "vocabulary.txt",
                       pos_file = "pos.txt",
                       ne_file = "ne.txt"):
    data_split_list = ["train", "development", "test"]
    
    # Read all raw data
    raw_data = {}
    for split in data_split_list:
        full_path = os.path.join(raw_data_path, split, "data/english/annotations")
        config = {"file_suffix": "gold_conll", "dir_prefix": full_path}
        raw_data[split] = load_data(config)
    
    # Read word list
    word_list, word_to_index = read_vocabulary(vocabulary_file)
    
    # Read POS list
    pos_list, pos_to_index = read_pos(pos_file)
    
    # Read named entity list
    ne_list, ne_to_index = read_ne(ne_file)
    
    # Build a tree structure for each sentence
    data = {}
    max_degree = 0
    pos_count = {}
    ne_count = {}
    pos_ne_count = {}
    for split in data_split_list:
        tree_data, degree, pos_count[split], ne_count[split], pos_ne_count[split] = (
            get_tree_data(raw_data[split], word_to_index, pos_to_index))
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
    
    # Compute the mapping to labels
    ne_to_index["NONE"] = len(ne_to_index)
    
    # Add label to nodes
    for split in data_split_list:
        for root_node in data[split][0]:
            label_tree_data(root_node, pos_to_index, ne_to_index)
    
    return data, max_degree, word_to_index, len(ne_to_index), len(pos_to_index), ne_list
    
def get_formatted_input(root_node, degree):
    """ Get inputs with RNN required format
    
    y1: vector; ne labels of nodes
    y2: vector; pos labels of nodes
    T: matrix; the tree structure
    p: vector; pos indices of nodes
    x1: vector; word indices of nodes
    x2: matrix; head word indices + parent head word indices + neighbor word indices
    S: matrix; [sibling]*siblings + [self] + [sibling]*siblings indices of nodes
    """
    # Get BFS layers
    layer_list = []
    layer = [root_node]
    while layer:
        layer_list.append(layer)
        child_layer = []
        for node in layer:
            child_layer.extend(node.child_list)
        layer = child_layer
    
    # Extract data from layers bottom-up
    y1 = []
    y2 = []
    T = []
    p = []
    x1 = []
    x2 = []
    
    S_tmp = []
    siblings = 1
    
    index = -1
    for layer in reversed(layer_list):
        for node in layer:
            index += 1
            node.index = index
            
            y1.append(node.y1)
            y2.append(node.y2)
            
            child_index_list = [child.index for child in node.child_list]
            T.append(child_index_list + [-1]*(degree-len(node.child_list)))
            
            p.append(node.pos_index)
            x1.append(node.word_index)
            x2.append([node.head_index, node.parent_head_index] + node.window_index_list)
            
            S_tmp.append([-1]*siblings + child_index_list + [-1]*siblings)
            
    y1 = np.array(y1, dtype=np.int32)
    y2 = np.array(y2, dtype=np.int32)
    T = np.array(T, dtype=np.int32)
    p = np.array(p, dtype=np.int32)
    x1 = np.array(x1, dtype=np.int32)
    x2 = np.array(x2, dtype=np.int32)
    
    S = np.zeros((len(y1), 2*siblings+1), dtype=np.int32)
    for sibling_list in S_tmp:
        for i in range(siblings, len(sibling_list)-siblings):
            S[sibling_list[i]] = sibling_list[i-siblings:i+siblings+1]
    
    return y1, y2, T, p, x1, x2, S

def get_one_hot(a, dimension):
    samples = len(a)
    A = np.zeros((samples, dimension), dtype=np.float32)
    A[np.arange(samples), a] = 1
    A = A * np.not_equal(a,-1).reshape((samples,1))
    return A

if __name__ == "__main__":
    # extract_glove_embeddings()
    # extract_conll_vocabulary()
    read_conll_dataset()
    exit()
    
    
    
    
    
