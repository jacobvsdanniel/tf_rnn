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
        self.child_list = []
        self.parent = None
        self.left = None
        self.right = None
        
    def add_child(self, child):
        if self.child_list:
            sibling = self.child_list[-1]
            sibling.right = child
            child.left = sibling
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
                                # character_to_index, word_to_index, pos_to_index,
                                # pos_count, ne_count, pos_ne_count)
    pos = tree.label
    word = tree.word
    span = tree.span
    head = tree.head if hasattr(tree, "head") else head_raw_data[(span, pos)][1]
    ne = ner_raw_data[span] if span in ner_raw_data else "NONE"
    # if ne in ["WORK_OF_ART", "FAC"]: ne = "NONE"
    
    # Process pos info
    node.pos = pos
    node.pos_index = pos_to_index[pos]
    pos_count[pos] += 1
    
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
    
    for document in raw_data["auto"]:
        for part in raw_data["auto"][document]:
            
            ner_raw_data = defaultdict(lambda: {})
            for k, v in raw_data["gold"][document][part]["ner"].iteritems():
                ner_raw_data[k[0]][(k[1], k[2])] = v
            
            for index, parse in enumerate(raw_data["auto"][document][part]["parses"]):
                if parse.subtrees[0].label == "NOPARSE": continue
                
                head_raw_data = raw_data["auto"][document][part]["heads"][index]
                text_raw_data = raw_data["auto"][document][part]["text"][index]
                
                root_node = Node()
                (degree, nodes
                    ) = construct_node(
                            root_node, parse, ner_raw_data[index], head_raw_data, text_raw_data,
                            character_to_index, word_to_index, pos_to_index,
                            pos_count, ne_count, pos_ne_count)
                    
                max_degree = max(max_degree, degree)
                root_node.nodes = nodes
                                        
                root_list.append(root_node)
                ner_list.append(ner_raw_data[index])
                
    log(" %d sentences\n" % len(root_list))
    return root_list, max_degree, pos_count, ne_count, pos_ne_count, ner_list

def label_tree_data(node, pos_to_index, ne_to_index):
    node.y = ne_to_index[node.ne]
        
    for child in node.child_list:
        label_tree_data(child, pos_to_index, ne_to_index)
    return
    
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

def get_padded_word(word, word_length=20):
    word_cut = [-1] + word[:word_length-2] + [-2]
    padding = [-3] * (word_length - len(word_cut))
    return word_cut + padding

def get_formatted_input(root_node, degree):
    """ Get inputs with RNN required format
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
    
    # Index nodes bottom-up
    index = -1
    for layer in reversed(layer_list):
        for node in layer:
            index += 1
            node.index = index
    
    # Extract data from layers bottom-up
    e = []
    y = []
    T = []
    p = []
    x = []
    w = []
    S = []
    chunk = []
    l = 0
    for layer in reversed(layer_list):
        for node in layer:
            e.append(1)
            
            y.append(node.y)
            
            child_index_list = [child.index for child in node.child_list]
            T.append(child_index_list
                     + [-1] * (degree-len(node.child_list))
                     + [node.left.index if node.left else -1,
                        node.right.index if node.right else -1,
                        node.parent.index if node.parent else -1])
            
            p.append([node.pos_index,
                      node.left.pos_index if node.left else -1,
                      node.right.pos_index if node.right else -1])
            
            x.append([node.word_index,
                      node.head_index,
                      node.left.head_index if node.left else -1,
                      node.right.head_index if node.right else -1])
            
            w.append([get_padded_word(node.word_split),
                      get_padded_word(node.head_split),
                      get_padded_word(node.left.head_split if node.left else []),
                      get_padded_word(node.right.head_split if node.right else [])])
            
            # S.append([node.index,
                      # node.left.index if node.left else -1,
                      # node.right.index if node.right else -1,
                      # node.parent.index if node.parent else -1])
            S.append([node.index,
                      node.left.index if node.left else -1,
                      node.right.index if node.right else -1])
            # S.append([node.index,
                      # node.parent.index if node.parent else -1])
            # S.append([node.index])
            
            chunk.append(node.span)
    
            if node.word_index != -1: l += 1
                    
    e = np.array(e, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    T = np.array(T, dtype=np.int32)
    p = np.array(p, dtype=np.int32)
    x = np.array(x, dtype=np.int32)
    w = np.array(w, dtype=np.int32)
    S = np.array(S, dtype=np.int32)
    return e, y, T, p, x, w, S, chunk, l
            
def get_batch_input(root_list, degree):
    input_list = []
    for root_node in root_list:
        input_list.append(get_formatted_input(root_node, degree))
    
    samples = len(input_list)
    nodes = max([i[1].shape[0] for i in input_list])
    poses = 3
    words = 4
    neighbors = 3
    word_length = 20
    
    e =  0 * np.ones([nodes, samples                    ], dtype=np.float32)
    y = -1 * np.ones([nodes, samples                    ], dtype=np.int32)
    T = -1 * np.ones([nodes, samples, degree+3          ], dtype=np.int32)
    p = -1 * np.ones([nodes, samples, poses             ], dtype=np.int32)
    x = -1 * np.ones([nodes, samples, words             ], dtype=np.int32)
    w = -3 * np.ones([nodes, samples, words, word_length], dtype=np.int32)
    S = -1 * np.ones([nodes, samples, neighbors         ], dtype=np.int32)
    chunk = []
    l =  np.zeros(samples, dtype=np.float32)
    
    for sample, i in enumerate(input_list):
        n = i[1].shape[0]
        e[:n, sample      ] = i[0]
        y[:n, sample      ] = i[1]
        T[:n, sample, :   ] = i[2]
        p[:n, sample, :   ] = i[3]
        x[:n, sample, :   ] = i[4]
        w[:n, sample, :, :] = i[5]
        S[:n, sample, :   ] = i[6]
        chunk.append(i[7])
        l[sample] = i[8]
    return e, y, T, p, x, w, S, chunk, l
    
if __name__ == "__main__":
    # extract_conll_vocabulary()
    # extract_collobert_embeddings()
    # extract_glove_embeddings()
    read_conll_dataset()
    exit()
    
    
    
    
    
