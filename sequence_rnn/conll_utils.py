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
        self.child = None
        self.parent = None
        
    def add_child(self, child):
        self.child = child
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
    
    ne_list = ["NOT_NE"]
    with open(ne_file, "r") as f:
        for line in f.readlines():
            ne = line.strip()
            ne_list += [ne+"_head", ne+"_body"]
            # ne_list.append(ne)
            # ne_list += [ne+"_single", ne+"_head", ne+"_body", ne+"_tail"]
            
    ne_to_index = {}
    for index, ne in enumerate(ne_list):
        ne_to_index[ne] = index
    
    log(" %d ne\n" % len(ne_to_index))
    return ne_list, ne_to_index

def construct_node(tail, tree, head_raw_data, ner, span_set,
                    character_to_index, word_to_index, pos_to_index, ne_to_index):
    span_set.add(tree.span)
    
    if tree.word:
        node = Node()
        node.pos_index = pos_to_index[tree.label]
        node.word_index = word_to_index[tree.word]
        node.text_index = tree.span[0]
        node.ne_index = ne_to_index[ner[node.text_index]]
        tail.add_child(node)
        tail = node
        
    # Binarize children
    degree = len(tree.subtrees)
    if degree > 2:
        head = tree.head if hasattr(tree, "head") else head_raw_data[(tree.span, tree.label)][1]
        side_child_pos = tree.subtrees[-1].label
        side_child_span = tree.subtrees[-1].span
        side_child_head = head_raw_data[(side_child_span, side_child_pos)][1]
        if side_child_head != head:
            sub_subtrees = tree.subtrees[:-1]
        else:
            sub_subtrees = tree.subtrees[1:]
        new_span = (sub_subtrees[0].span[0], sub_subtrees[-1].span[1])
        new_tree = PSTree(label=tree.label, span=new_span, subtrees=sub_subtrees)
        new_tree.head = head
        if side_child_head != head:
            tree.subtrees = [new_tree, tree.subtrees[-1]]
        else:
            tree.subtrees = [tree.subtrees[0], new_tree]
      
    # Process children
    for subtree in tree.subtrees:
        tail = construct_node(tail, subtree, head_raw_data, ner, span_set,
                character_to_index, word_to_index, pos_to_index, ne_to_index)
    return tail

def get_tree_data(raw_data, character_to_index, word_to_index, pos_to_index, ne_to_index):
    log("get_tree_data()...")
    root_list = []
    ner_list = []
    span_set_list = []
    
    for document in raw_data["auto"]:
        for part in raw_data["auto"][document]:
            
            ner_raw_data = defaultdict(lambda: {})
            for k, v in raw_data["gold"][document][part]["ner"].iteritems():
                ner_raw_data[k[0]][(k[1], k[2])] = v
            
            for index, parse in enumerate(raw_data["auto"][document][part]["parses"]):
                if parse.subtrees[0].label == "NOPARSE": continue
                
                text_raw_data = raw_data["auto"][document][part]["text"][index]
                sentence_length = len(text_raw_data)
                ner = ["NOT_NE"] * sentence_length
                for chunk, ne in ner_raw_data[index].iteritems():
                    ner[chunk[0]] = ne + "_head"
                    for i in range(chunk[0]+1, chunk[1]):
                        ner[i] = ne + "_body"
                    """
                    for i in range(chunk[0], chunk[1]):
                        ner[i] = ne
                    """
                    """
                    head = chunk[0]
                    tail = chunk[1] - 1
                    if head == tail:
                        ner[head] = ne + "_single"
                    else:
                        ner[head] = ne + "_head"
                        ner[tail] = ne + "_tail"
                        for i in range(head+1, tail):
                            ner[i] = ne + "_body"
                    """
                
                head_raw_data = raw_data["auto"][document][part]["heads"][index]
                span_set = set()
                
                root_node = Node()
                _ = construct_node(root_node, parse, head_raw_data, ner, span_set,
                        character_to_index, word_to_index, pos_to_index, ne_to_index)
                
                root_node.nodes = sentence_length
                root_node.text = text_raw_data
                        
                root_list.append(root_node)
                ner_list.append(ner_raw_data[index])
                span_set_list.append(span_set)
    
    sentences = len(root_list)
    nodes = sum(root_node.nodes for root_node in root_list)
    nes = sum(len(ner) for ner in ner_list)
    log(" %d sentences; %d nodes; %d named entities\n" % (sentences, nodes, nes))
    return root_list, ner_list, span_set_list
   
def read_conll_dataset(data_split_list = ["train", "development", "test"]):
    character_file = "character.txt"
    vocabulary_file = "vocabulary.txt"
    pos_file = "pos.txt"
    ne_file = "ne.txt"
    data_path = "/home/danniel/Desktop/CONLL2012-intern/conll-2012/v4/data"
    test_auto_data_path = "/home/danniel/Downloads/wu_conll_test/v9/data"
    data_path_suffix = "data/english/annotations"
    annotation_method_list = ["gold", "auto"]
    
    # Read all raw data
    raw_data = {}
    for split in data_split_list:
        raw_data[split] = {}
        for method in annotation_method_list:
            if split == "test" and method == "auto":
                full_path = os.path.join(test_auto_data_path, split, data_path_suffix)
            else:
                full_path = os.path.join(data_path, split, data_path_suffix)
            config = {"file_suffix": "%s_conll" % method, "dir_prefix": full_path}
            raw_data[split][method] = load_data(config)
    
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
    for split in data_split_list:
        log("<%s>\n " % split)
        root_list, ner_list, span_data = (
            get_tree_data(raw_data[split],
                character_to_index, word_to_index, pos_to_index, ne_to_index))
        data[split] = [root_list, ner_list, span_data]
    return (data, word_to_index,
            len(ne_to_index), len(pos_to_index), len(character_to_index),
            ne_list)

def get_padded_word(word, word_length=20):
    word_cut = [-1] + word[:word_length-2] + [-2]
    padding = [-3] * (word_length - len(word_cut))
    return word_cut + padding

def get_formatted_input(root_node):
    """ Get inputs with RNN required format
    """
    y = []
    p = []
    x = []
    node = root_node
    while node.child:
        node = node.child
        y.append(node.ne_index)
        p.append([node.pos_index])
        x.append([node.word_index])
            
    y = np.array(y, dtype=np.int32)
    p = np.array(p, dtype=np.int32)
    x = np.array(x, dtype=np.int32)
    e = np.ones_like(y, dtype=np.float32)
    return y, p, x, e
            
def get_batch_input(root_list):
    input_list = []
    for root_node in root_list:
        input_list.append(get_formatted_input(root_node))
    
    samples = len(input_list)
    nodes = max([y.shape[0] for y, p, x, e in input_list])
    poses = 1
    words = 1
    
    Y = -1 * np.ones([nodes, samples                    ], dtype=np.int32)
    P = -1 * np.ones([nodes, samples, poses             ], dtype=np.int32)
    X = -1 * np.ones([nodes, samples, words             ], dtype=np.int32)
    E = np.zeros([nodes, samples], dtype=np.float32)
    
    for sample, i in enumerate(input_list):
        y, p, x, e = i
        n = y.shape[0]
        Y[:n, sample      ] = y
        P[:n, sample, :   ] = p
        X[:n, sample, :   ] = x
        E[:n, sample      ] = e
    return Y, P, X, E
    
if __name__ == "__main__":
    # extract_conll_vocabulary()
    # extract_collobert_embeddings()
    # extract_glove_embeddings()
    read_conll_dataset()
    exit()
    
    
    
    
    
