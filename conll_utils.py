import re
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
        self.parent = self
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

def get_mapped_word(word):
    if not word: return word, -1
    case = get_case_info(word)
    # word = word.lower()
    # word = re.sub("\d+", "NUMBER", word.lower())
    return word, case

def get_case_info(word):
    if word.isupper():
        return 0
    if word.islower():
        return 1
    if word[0].isupper():
        return 2
    else:
        return 3

def extract_collobert_embeddings(
        embedding_file = "/home/danniel/Downloads/senna/embeddings/embeddings.txt",
        word_file = "/home/danniel/Downloads/senna/hash/words.lst",
        vocabulary_file = "vocabulary.txt"):
    
    word_list, word_to_index = read_vocabulary(vocabulary_file)
    
    word_list = []
    embedding_list = []
    with open(word_file, "r") as fw, open(embedding_file, "r") as fe:
        while True:
            word = fw.readline().strip()
            embedding = fe.readline().strip().split()
            if not word: break
            if word not in word_to_index: continue
            embedding = np.array([float(i) for i in embedding])
            word_list.append(word)
            embedding_list.append(embedding)
    
    np.save("collobert_word.npy", word_list)
    np.save("collobert_embedding.npy", embedding_list)
    return
        
def extract_glove_embeddings(glove_file = "../glove.840B.300d.txt",
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
                             vocabulary_file = "vocabulary.txt",
                             character_file = "character.txt"):
    log("extract_conll_vocabulary()...")
    
    character_set = set()
    vocabulary_set = set()
    for split in ["train", "development", "test"]:
        full_path = os.path.join(raw_data_path, split, "data/english/annotations")
        config = {"file_suffix": "gold_conll", "dir_prefix": full_path}
        raw_data = load_data(config)
        for document in raw_data:
            for part in raw_data[document]:
                for sentence in raw_data[document][part]["text"]:
                    for word in sentence:
                        for character in word:
                            character_set.add(character)
                        vocabulary_set.add(get_mapped_word(word)[0])
    
    with codecs.open(vocabulary_file, "w", encoding="utf8") as f:
        for word in sorted(vocabulary_set):
            f.write(word + '\n')
    
    with codecs.open(character_file, "w", encoding="utf8") as f:
        for character in sorted(character_set):
            f.write(character + '\n')
    
    log(" done\n")
    return

def read_character(character_file):
    log("read_character()...")
    
    character_list = []
    character_to_index = {}
    with codecs.open(character_file, "r", encoding="utf8") as f:
        for line in f.readlines():
            character = line[0]
            character_to_index[character] = len(character_list)
            character_list.append(character)
    
    log(" %d characters\n" % len(character_to_index))
    return character_list, character_to_index

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

def construct_node(node, tree, ner_raw_data, head_raw_data, text_raw_data,
                    character_to_index, word_to_index, pos_to_index,
                    pos_count, ne_count, pos_ne_count):
    # if len(tree.subtrees) == 1:
        # return construct_node(node, tree.subtrees[0], ner_raw_data, head_raw_data, text_raw_data,
                                # word_to_index, pos_to_index,
                                # pos_count, ne_count, pos_ne_count)
    pos = tree.label
    word = tree.word
    span = tree.span
    head = tree.head if hasattr(tree, "head") else head_raw_data[(span, pos)][1]
    ne = ner_raw_data[span] if span in ner_raw_data else "NONE"
    
    # Process pos info
    node.pos = pos
    node.pos_index = pos_to_index[pos]
    pos_count[pos] += 1
    # if not word: node.pos_index = -1
    
    # Process word info
    node.word_split = [character_to_index[character] for character in word] if word else []
    mapped_word, node.word_case = get_mapped_word(word)
    node.word_index = word_to_index[mapped_word] if mapped_word else -1
    
    # Process head info
    node.head_split = [character_to_index[character] for character in head]
    mapped_head, node.head_case = get_mapped_word(head)
    node.head_index = word_to_index[mapped_head]
    
    # Process ne info
    node.ne = ne
    ne_count[ne] += 1
    if ne != "NONE": pos_ne_count[pos] += 1
    
    # Process chunk info
    node.span = span
    
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
    node.degree = len(tree.subtrees)
    max_degree = node.degree
    nodes = 1
    for subtree in tree.subtrees:
        child = Node()
        node.add_child(child)
        (child_degree, child_nodes
            ) = construct_node(child, subtree, ner_raw_data, head_raw_data, text_raw_data,
                                character_to_index, word_to_index, pos_to_index,
                                pos_count, ne_count, pos_ne_count)
        max_degree = max(max_degree, child_degree)
        nodes += child_nodes
        
    return max_degree, nodes

def get_tree_data(raw_data, character_to_index, word_to_index, pos_to_index):
    log("get_tree_data()...")
    """ Get tree structured data from CoNLL 2012
    
    Stores into Node data structure
    """
    root_list = []
    ner_list = []
    max_degree = 0
    pos_count = defaultdict(lambda: 0)
    ne_count = defaultdict(lambda: 0)
    pos_ne_count = defaultdict(lambda: 0)
    
    for document in raw_data:
        for part in raw_data[document]:
            ner_raw_data = defaultdict(lambda: {})
            for k, v in raw_data[document][part]["ner"].iteritems():
                ner_raw_data[k[0]][(k[1], k[2])] = v
            
            for index, parse in enumerate(raw_data[document][part]["parses"]):
                head_raw_data = raw_data[document][part]["heads"][index]
                text_raw_data = raw_data[document][part]["text"][index]
                
                root_node = Node()
                (degree, nodes
                    ) = construct_node(
                            root_node, parse, ner_raw_data[index], head_raw_data, text_raw_data,
                            character_to_index, word_to_index, pos_to_index,
                            pos_count, ne_count, pos_ne_count)
                max_degree = max(max_degree, degree)
                root_node.nodes = nodes
                # root_node.text = raw_data[document][part]["text"][index]
                                        
                root_list.append(root_node)
                ner_list.append(ner_raw_data[index])
                
    log(" %d sentences\n" % len(root_list))
    return root_list, max_degree, pos_count, ne_count, pos_ne_count, ner_list

def label_tree_data(node, pos_to_index, ne_to_index):
    node.y1 = ne_to_index[node.ne]
        
    for child in node.child_list:
        label_tree_data(child, pos_to_index, ne_to_index)
    return
    
def read_conll_dataset(raw_data_path = "../CONLL2012-intern/conll-2012/v4/data",
                       character_file = "character.txt",
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
    
    # Read character list
    character_list, character_to_index = read_character(character_file)
    
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
        tree_data, degree, pos_count[split], ne_count[split], pos_ne_count[split], ner_list = (
            get_tree_data(raw_data[split], character_to_index, word_to_index, pos_to_index))
        sentences = len(tree_data)
        nodes = sum(pos_count[split].itervalues())
        nes = sum(pos_ne_count[split].itervalues())
        max_degree = max(max_degree, degree)
        data[split] = [tree_data, nodes, nes, ner_list]
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
    
    # Real total nes
    reals = 0
    for split in data_split_list:
        for ner_dict in data[split][3]:
            reals += len(ner_dict)
    print "\n[Real] Total %d named entities" % reals
        
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
    
    return (data, max_degree, word_to_index,
            len(ne_to_index), len(pos_to_index), len(character_to_index),
            ne_list)

def get_padded_word(word, word_length):
    word_cut = [-1] + word[:word_length-2] + [-2]
    padding = [-3] * (word_length - len(word_cut))
    return word_cut + padding

def get_formatted_input(root_node, degree, word_length):
    """ Get inputs with RNN required format
    
    y1: vector; ne labels of nodes
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
    T = []
    p = []
    
    x1 = []
    x2 = []
    x3 = []
    xx = []
    
    chunk = []
    S_tmp = []
    siblings = 1
    
    index = -1
    for layer in reversed(layer_list):
        for node in layer:
            index += 1
            node.index = index
                
            y1.append(node.y1)
            
            child_index_list = [child.index for child in node.child_list]
            T.append(child_index_list + [-1]*(degree-len(node.child_list)))
            
            p.append(node.pos_index)
            
            x1.append(node.word_index)
            x2.append(node.head_index)
            x3.append(node.parent.head_index)
            
            xx.append(get_padded_word(node.word_split, word_length))
            xx.append(get_padded_word(node.head_split, word_length))
            xx.append(get_padded_word(node.parent.head_split, word_length))
            
            chunk.append(node.span)
            S_tmp.append([-1]*siblings + child_index_list + [-1]*siblings)
            
    y1 = np.array(y1, dtype=np.int32)
    T = np.array(T, dtype=np.int32)
    p = np.array(p, dtype=np.int32)
    
    x1 = np.array(x1, dtype=np.int32)
    x2 = np.array(x2, dtype=np.int32)
    x3 = np.array(x3, dtype=np.int32)
    xx = np.array(xx, dtype=np.int32)
    
    S = np.ones((len(y1), 2*siblings+1), dtype=np.int32) * -1
    # S = np.ones((len(y1), 2*siblings+2), dtype=np.int32) * -1
    for index, child_index_list in enumerate(S_tmp):
        for i in range(siblings, len(child_index_list)-siblings):
            S[child_index_list[i]] = child_index_list[i-siblings:i+siblings+1]
            # S[child_index_list[i],:-1] = child_index_list[i-siblings:i+siblings+1]
            # S[child_index_list[i], -1] = index
    S[-1,1] = len(y1) - 1
    
    return (y1, T, p, 
            x1, x2, x3, xx,
            S, chunk)
            
def get_batch_input(root_list, degree, word_length=20):
    input_list = []
    for root_node in root_list:
        # y1, T, p, x1, x2, x3, xx, S, chunk = get_formatted_input(root_node, degree)
        input_list.append(get_formatted_input(root_node, degree, word_length))
    
    samples = len(input_list)
    nodes = max([inp[1].shape[0] for inp in input_list])
    
    y1 = -1 * np.ones([samples, nodes], dtype=np.int32)
    T = -1 * np.ones([samples, nodes, degree], dtype=np.int32)
    p = -1 * np.ones([samples, nodes], dtype=np.int32)
    x1 = -1 * np.ones([samples, nodes], dtype=np.int32)
    x2 = -1 * np.ones([samples, nodes], dtype=np.int32)
    x3 = -1 * np.ones([samples, nodes], dtype=np.int32)
    xx = -3 * np.ones([samples, nodes*3, word_length], dtype=np.int32)
    S = -1 * np.ones([samples, nodes, 3], dtype=np.int32)
    chunk = []
    
    for i, inp in enumerate(input_list):
        y1[i, :inp[0].shape[0]] = inp[0]
        T[i, :inp[1].shape[0], :] = inp[1]
        p[i, :inp[2].shape[0]] = inp[2]
        x1[i, :inp[3].shape[0]] = inp[3]
        x2[i, :inp[4].shape[0]] = inp[4]
        x3[i, :inp[5].shape[0]] = inp[5]
        xx[i, :inp[6].shape[0], :] = inp[6]
        S[i, :inp[7].shape[0], :] = inp[7]
        chunk.append(inp[8])
    
    return (y1, T, p, 
            x1, x2, x3, xx,
            S, chunk)
    
def get_one_hot(a, dimension):
    samples = len(a)
    A = np.zeros((samples, dimension), dtype=np.float32)
    A[np.arange(samples), a] = 1
    A = A * np.not_equal(a,-1).reshape((samples,1))
    return A

if __name__ == "__main__":
    # extract_conll_vocabulary()
    # extract_collobert_embeddings()
    # extract_glove_embeddings()
    read_conll_dataset()
    exit()
    
    
    
    
    
