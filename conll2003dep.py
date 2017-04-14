import re
import os
import sys
import time
import codecs
import difflib
import subprocess
from collections import defaultdict

import numpy as np

sys.path.append("/home/danniel/Desktop/CONLL2012-intern")
import pstree
import head_finder

from rnn import Node
import dependency_utils

dataset = "conll2003dep"
character_file = os.path.join(dataset, "character.txt")
word_file = os.path.join(dataset, "word.txt")
pos_file = os.path.join(dataset, "pos.txt")
ne_file = os.path.join(dataset, "ne.txt")
pretrained_word_file = os.path.join(dataset, "word.npy")
pretrained_embedding_file = os.path.join(dataset, "embedding.npy")

lexicon_phrase_file = os.path.join(dataset, "lexicon_phrase.npy")
lexicon_embedding_file = os.path.join(dataset, "lexicon_embedding.npy")
senna_path = "/home/danniel/Downloads/senna/hash"
dbpedia_path = "/home/danniel/Desktop/dbpedia_lexicon"
lexicon_meta_list = [
    {"ne": "PER",  "path": os.path.join(dataset, "senna_per.txt"),    "senna": os.path.join(senna_path, "ner.per.lst")}, 
    {"ne": "ORG",  "path": os.path.join(dataset, "senna_org.txt"),    "senna": os.path.join(senna_path, "ner.org.lst")},
    {"ne": "LOC",  "path": os.path.join(dataset, "senna_loc.txt"),    "senna": os.path.join(senna_path, "ner.loc.lst")},
    {"ne": "MISC", "path": os.path.join(dataset, "senna_misc.txt"),   "senna": os.path.join(senna_path, "ner.misc.lst")},
    {"ne": "PER",  "path": os.path.join(dataset, "dbpedia_per.txt"),  "dbpedia": os.path.join(dbpedia_path, "dbpedia_person.txt")},
    {"ne": "ORG",  "path": os.path.join(dataset, "dbpedia_org.txt"),  "dbpedia": os.path.join(dbpedia_path, "dbpedia_organisation.txt")},
    {"ne": "LOC",  "path": os.path.join(dataset, "dbpedia_loc.txt"),  "dbpedia": os.path.join(dbpedia_path, "dbpedia_place.txt")},
    {"ne": "MISC", "path": os.path.join(dataset, "dbpedia_misc.txt"), "dbpedia": os.path.join(dbpedia_path, "dbpedia_work.txt")}]

project_path = "/home/danniel/Desktop/rnn_ner"
parse_script = os.path.join(project_path, dataset, "parse.sh")

data_path = "/home/danniel/Downloads/ner"
glove_file = "/home/danniel/Downloads/glove.840B.300d.txt"
syntaxnet_path = "/home/danniel/Downloads/tf_models/syntaxnet"

split_raw = {"train": "eng.train", "validate": "eng.testa", "test": "eng.testb"}
split_sentence = {"train": "sentence_train.conllu", "validate": "sentence_validate.conllu", "test": "sentence_test.conllu"}
split_dependency = {"train": "dependency_train.conllu", "validate": "dependency_validate.conllu", "test": "dependency_test.conllu"}
split_constituency = {"train": "constituency_train.txt", "validate": "constituency_validate.txt", "test": "constituency_test.txt"}

def log(msg):
    sys.stdout.write(msg)
    sys.stdout.flush()
    return

def read_list_file(file_path, encoding="utf8"):
    log("Read %s..." % file_path)
    
    with codecs.open(file_path, "r", encoding=encoding) as f:
        line_list = f.read().splitlines()
    line_to_index = {line: index for index, line in enumerate(line_list)}
    
    log(" %d lines\n" % len(line_to_index))
    return line_list, line_to_index

def group_sequential_label(seq_ne_list):
    span_ne_dict = {}
    
    start, ne = -1, None
    for index, label in enumerate(seq_ne_list + ["O"]):
        if (label[0]=="O" or label[0]=="B") and ne:
            span_ne_dict[(start, index)] = ne
            start, ne = -1, None
        
        if label[0]=="B" or (label[0]=="I" and not ne):
            start, ne = index, label[2:]
        
    return span_ne_dict

def extract_ner(split):
    with open(os.path.join(data_path, split_raw[split]), "r") as f:
        line_list = f.read().splitlines()
    
    sentence_list = []
    ner_list = []
    
    sentence = []
    ner = []
    for line in line_list[2:]:
        if line[:10] == "-DOCSTART-": continue
        if not line:
            if sentence:
                sentence_list.append(sentence)
                ner_list.append(group_sequential_label(ner))
                sentence = []
                ner = []
            continue
        word, _, _, sequential_label = line.split()
        sentence.append(word)
        ner.append(sequential_label)
    
    return sentence_list, ner_list
"""
def get_parse_tree(parse_string, pos_set=None):
    node = Node()
    
    # get POS
    header, parse_string = parse_string.split(" ", 1)
    node.pos = header[1:]
    if pos_set is not None: pos_set.add(node.pos)
    
    # bottom condition: hit a word
    if parse_string[0] != "(":
        node.word, parse_string = parse_string.split(")", 1)
        return node, parse_string
    node.word = None
    #node.word = ""
    
    # Process children
    while True:
        child, parse_string = get_parse_tree(parse_string, pos_set)
        node.add_child(child)
        delimiter, parse_string = parse_string[0], parse_string[1:]
        if delimiter == ")":
            return node, parse_string

def print_parse_tree(node, indent):
    print indent, node.pos, node.word
    for child in node.child_list:
        print_parse_tree(child, indent+"    ")
    return
"""
def extract_pos_from_tree(tree, pos_set):
    pos_set.add(tree.pos)
    
    for child in tree.child_list:
        extract_pos_from_tree(child, pos_set)
    return

def prepare_dataset():
    ne_set = set()
    word_set = set()
    character_set = set()
    pos_set = set()
    
    for split in split_raw:
        sentence_list, ner_list = extract_ner(split)
        
        # Procecss raw NER
        for ner in ner_list:
            for ne in ner.itervalues():
                ne_set.add(ne)
        
        # Procecss raw sentences and store into conllu format
        sentence_file = os.path.join(dataset, split_sentence[split])
        with open(sentence_file, "w") as f:
            for sentence in sentence_list:
                f.write("#" + " ".join(sentence) + "\n")
                for i, word in enumerate(sentence):
                    f.write("%d\t"%(i+1) + word + "\t_"*8 + "\n")
                    word_set.add(word)
                    for character in word:
                        character_set.add(character)
                f.write("\n")
        
        # Generate dependency parses
        subprocess.call([parse_script, split], cwd=syntaxnet_path)
        
        # Transform dependency parses to constituency parses
        dependency_file = os.path.join(dataset, split_dependency[split])
        dependency_list = dependency_utils.read_conllu(dependency_file)
        for dependency_parse in dependency_list:
            constituency_parse = dependency_utils.dependency_to_constituency(*dependency_parse)
            extract_pos_from_tree(constituency_parse, pos_set)
            
    with open(ne_file, "w") as f:
        for ne in sorted(ne_set):
            f.write(ne + '\n')
    
    with open(word_file, "w") as f:
        for word in sorted(word_set):
            f.write(word + '\n')
    
    with open(character_file, "w") as f:
        for character in sorted(character_set):
            f.write(character + '\n')
    
    with open(pos_file, "w") as f:
        for pos in sorted(pos_set):
            f.write(pos + '\n')
    return

def traverse_tree(node, ner_raw_data, text_raw_data, index_to_lexicon):
    node.ne = ner_raw_data[node.span] if node.span in ner_raw_data else "NONE"
    node.constituent = " ".join(text_raw_data[node.span[0]:node.span[1]]).lower()
    
    for index, lexicon in index_to_lexicon.iteritems():
        if node.constituent in lexicon and node.ne != lexicon_meta_list[index]["ne"]:
            del lexicon[node.constituent]
        #difflib.get_close_matches(node.constituent, lexicon[ne].iterkeys(), 1, 0.8)
        #all(difflib.SequenceMatcher(a=node.constituent, b=phrase).ratio() < 0.8 for phrase in lexicon[ne])
            
    # Process children
    for child in node.child_list:
        traverse_tree(child, ner_raw_data, text_raw_data, index_to_lexicon)
    return

def extract_clean_senna_lexicon():
    index_to_lexicon = {}
    
    print "\nReading raw lexicon from senna..."
    for index, meta in enumerate(lexicon_meta_list):
        if "senna" not in meta: continue
        _, index_to_lexicon[index] = read_list_file(meta["senna"], "iso8859-15")
    print "-"*50 + "\n   ne  phrases longest\n" + "-"*50
    for index, lexicon in index_to_lexicon.iteritems():
        longest_phrase = max(lexicon.iterkeys(), key=lambda phrase: len(phrase))
        print "%5s %8d %s" % (lexicon_meta_list[index]["ne"], len(lexicon), longest_phrase)
    
    log("\nReading training data...")
    data_split_list = ["train", "validate"]
    sentence_data = {}
    ner_data = {}
    parse_data = {}
    for split in data_split_list:
        sentence_data[split], ner_data[split] = extract_ner(split)
        
        dependency_file = os.path.join(dataset, split_dependency[split])
        dependency_parse_list = dependency_utils.read_conllu(dependency_file)
        parse_data[split] = [dependency_utils.dependency_to_constituency(*parse)
            for parse in dependency_parse_list]
    log(" done\n")
        
    log("\nCleaning lexicon by training data...")
    for split in data_split_list:
        for index, parse in enumerate(parse_data[split]):
            traverse_tree(parse, ner_data[split][index], sentence_data[split][index], index_to_lexicon)
    log(" done\n")
    print "-"*50 + "\n   ne  phrases longest\n" + "-"*50
    for index, lexicon in index_to_lexicon.iteritems():
        longest_phrase = max(lexicon.iterkeys(), key=lambda phrase: len(phrase))
        print "%5s %8d %s" % (lexicon_meta_list[index]["ne"], len(lexicon), longest_phrase)
        
    for index, meta in enumerate(lexicon_meta_list):
        with codecs.open(meta["path"], "w", encoding="iso8859-15") as f:
            for phrase in index_to_lexicon[index]:
                f.write("%s\n" % phrase)
    return
    
def extract_lexicon_embeddings():
    log("extract_lexicon_embeddings()...")
    
    # Read senna lexicon
    lexicon = defaultdict(lambda: [0]*len(lexicon_meta_list))
    for index, meta in enumerate(lexicon_meta_list):
        for phrase in read_list_file(meta["path"], "iso8859-15")[0]:
            lexicon[phrase][index] = 1
    
    # Create embeddings
    phrase_list, embedding_list = zip(*lexicon.iteritems())
    np.save(lexicon_phrase_file, phrase_list)
    np.save(lexicon_embedding_file, embedding_list)
    
    log(" %d phrases in lexicon\n" % len(phrase_list))
    return

def extract_glove_embeddings():
    log("extract_glove_embeddings()...")
    
    _, word_to_index = read_list_file(word_file)
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
    
    np.save(pretrained_word_file, word_list)
    np.save(pretrained_embedding_file, embedding_list)
    
    log(" %d pre-trained words\n" % len(word_list))
    return

def construct_node(node, ner_raw_data, text_raw_data,
        character_to_index, word_to_index, pos_to_index, index_to_lexicon,
        pos_count, ne_count, pos_ne_count, lexicon_hits, span_to_node):
    pos = node.pos
    word = node.word
    head = node.head
    span = node.span
    ne = ner_raw_data[span] if span in ner_raw_data else "NONE"
    constituent = " ".join(text_raw_data[span[0]:span[1]]).lower()
    
    # Process pos info
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
    if ne != "NONE":
        if not node.parent or node.parent.span!=span:
            ne_count[ne] += 1
        pos_ne_count[pos] += 1
    
    # Process span info
    span_to_node[span] = node
    
    # Process lexicon info
    node.lexicon_hit = [0] * len(index_to_lexicon)
    hits = 0
    for index, lexicon in index_to_lexicon.iteritems():
        if constituent in lexicon:
            #lexicon[constituent] += 1
            node.lexicon_hit[index] = 1
            hits = 1
    lexicon_hits[0] += hits
    
    # Process children
    nodes = 1
    for child in node.child_list:
        child_nodes = construct_node(child, ner_raw_data, text_raw_data,
            character_to_index, word_to_index, pos_to_index, index_to_lexicon,
            pos_count, ne_count, pos_ne_count, lexicon_hits, span_to_node)
        nodes += child_nodes
    return nodes

def create_dense_nodes(ner_raw_data, text_raw_data, pos_to_index, index_to_lexicon,
        pos_count, ne_count, pos_ne_count, lexicon_hits, span_to_node):
    node_list = []
    max_dense_span = 3
    # Start from bigram, since all unigrams are already covered by parses
    for span_length in range(2, 1+max_dense_span):
        for span_start in range(0, 1+len(text_raw_data)-span_length):
            span = (span_start, span_start+span_length)
            if span in span_to_node: continue
            pos = "NONE"
            ne = ner_raw_data[span] if span in ner_raw_data else "NONE"
            constituent = " ".join(text_raw_data[span[0]:span[1]]).lower()
            
            # span, child
            # TODO: sibling
            node = Node()
            node_list.append(node)
            node.span = span
            span_to_node[span] = node
            node.child_list = [span_to_node[(span[0],span[1]-1)], span_to_node[(span[0]+1,span[1])]]
            
            # word, head, pos
            node.pos_index = pos_to_index[pos]
            pos_count[pos] += 1
            node.word_split = []
            node.word_index = -1
            node.head_split = []
            node.head_index = -1
            
            # ne
            node.ne = ne
            if ne != "NONE":
                ne_count[ne] += 1
                pos_ne_count[pos] += 1
            
            # lexicon
            node.lexicon_hit = [0] * len(index_to_lexicon)
            hits = 0
            for index, lexicon in index_to_lexicon.iteritems():
                if constituent in lexicon:
                    #lexicon[constituent] += 1
                    node.lexicon_hit[index] = 1
                    hits = 1
            lexicon_hits[0] += hits
    
    return node_list
    
def get_tree_data(sentence_list, parse_list, ner_list,
        character_to_index, word_to_index, pos_to_index, index_to_lexicon):
    log("get_tree_data()...")
    """ Get tree structured data from CoNLL-2003
    
    Stores into Node data structure
    """
    tree_pyramid_list = []
    word_count = 0
    pos_count = defaultdict(lambda: 0)
    ne_count = defaultdict(lambda: 0)
    pos_ne_count = defaultdict(lambda: 0)
    lexicon_hits = [0]
    
    for index, parse in enumerate(parse_list):
        text_raw_data = sentence_list[index]
        word_count += len(text_raw_data)
        span_to_node = {}
        
        nodes = construct_node(
            parse, ner_list[index], text_raw_data,
            character_to_index, word_to_index, pos_to_index, index_to_lexicon,
            pos_count, ne_count, pos_ne_count, lexicon_hits, span_to_node)
        parse.nodes = nodes
        parse.tokens = len(text_raw_data)
                
        #additional_node_list = []
        additional_node_list = create_dense_nodes(
            ner_list[index], text_raw_data,
            pos_to_index, index_to_lexicon,
            pos_count, ne_count, pos_ne_count, lexicon_hits, span_to_node)
                
        tree_pyramid_list.append((parse, additional_node_list))
        
    log(" %d sentences\n" % len(tree_pyramid_list))
    return tree_pyramid_list, word_count, pos_count, ne_count, pos_ne_count, lexicon_hits[0]

def label_tree_data(node, pos_to_index, ne_to_index):
    node.y = ne_to_index[node.ne]
        
    for child in node.child_list:
        label_tree_data(child, pos_to_index, ne_to_index)
    return
    
def read_dataset(data_split_list = ["train", "validate", "test"]):
    # Read all raw data
    sentence_data = {}
    ner_data = {}
    parse_data = {}
    for split in data_split_list:
        sentence_data[split], ner_data[split] = extract_ner(split)
        
        dependency_file = os.path.join(dataset, split_dependency[split])
        dependency_parse_list = dependency_utils.read_conllu(dependency_file)
        parse_data[split] = [dependency_utils.dependency_to_constituency(*parse)
            for parse in dependency_parse_list]
        
    # Read lists of annotations
    character_list, character_to_index = read_list_file(character_file)
    word_list, word_to_index = read_list_file(word_file)
    pos_list, pos_to_index = read_list_file(pos_file)
    ne_list, ne_to_index = read_list_file(ne_file)
    
    pos_to_index["NONE"] = len(pos_to_index)
    
    # Read lexicon
    index_to_lexicon = {}
    for index, meta in enumerate(lexicon_meta_list):
        #_, index_to_lexicon[index] = read_list_file(meta["path"], "iso8859-15")
        if "senna" in meta:
            _, index_to_lexicon[index] = read_list_file(meta["senna"], "iso8859-15")
        elif "dbpedia" in meta:
            _, index_to_lexicon[index] = read_list_file(meta["dbpedia"])
    """
    for index, lexicon in index_to_lexicon.iteritems():
        for name in lexicon:
            lexicon[name] = 0
    """
    # Build a tree structure for each sentence
    data = {}
    word_count = {}
    pos_count = {}
    ne_count = {}
    pos_ne_count = {}
    lexicon_hits = {}
    for split in data_split_list:
        (tree_pyramid_list,
            word_count[split], pos_count[split], ne_count[split], pos_ne_count[split],
            lexicon_hits[split]) = get_tree_data(
                sentence_data[split], parse_data[split], ner_data[split],
                character_to_index, word_to_index, pos_to_index, index_to_lexicon)
        data[split] = {"tree_pyramid_list": tree_pyramid_list, "ner_list": ner_data[split]}
    """
    for index, lexicon in index_to_lexicon.iteritems():
        with codecs.open("tmp_%d.txt" % index, "w", encoding="utf8") as f:
            for name, count in sorted(lexicon.iteritems(), key=lambda x: (-x[1], x[0])):
                if count == 0: break
                f.write("%9d %s\n" % (count, name))
    """
    # Show statistics of each data split 
    print "-" * 80
    print "%10s%10s%9s%9s%7s%12s%13s" % ("split", "sentence", "token", "node", "NE", "spanned_NE",
        "lexicon_hit")
    print "-" * 80
    for split in data_split_list:
        print "%10s%10d%9d%9d%7d%12d%13d" % (split,
            len(data[split]["tree_pyramid_list"]),
            word_count[split],
            sum(pos_count[split].itervalues()),
            sum(len(ner) for ner in data[split]["ner_list"]),
            sum(ne_count[split].itervalues()),
            lexicon_hits[split])
    
    # Show POS distribution
    total_pos_count = defaultdict(lambda: 0)
    for split in data_split_list:
        for pos in pos_count[split]:
            total_pos_count[pos] += pos_count[split][pos]
    nodes = sum(total_pos_count.itervalues())
    print "\nTotal %d nodes" % nodes
    print "-"*80 + "\n   POS   count  ratio\n" + "-"*80
    for pos, count in sorted(total_pos_count.iteritems(), key=lambda x: x[1], reverse=True)[:10]:
        print "%6s %7d %5.1f%%" % (pos, count, count*100./nodes)
    
    # Show NE distribution in [train, validate]
    total_ne_count = defaultdict(lambda: 0)
    for split in data_split_list:
        if split == "test": continue
        for ne in ne_count[split]:
            total_ne_count[ne] += ne_count[split][ne]
    nes = sum(total_ne_count.itervalues())
    print "\nTotal %d spanned named entities in [train, validate]" % nes
    print "-"*80 + "\n          NE  count  ratio\n" + "-"*80
    for ne, count in sorted(total_ne_count.iteritems(), key=lambda x: x[1], reverse=True):
        print "%12s %6d %5.1f%%" % (ne, count, count*100./nes)
    
    # Show POS-NE distribution in [train, validate]
    total_pos_ne_count = defaultdict(lambda: 0)
    for split in data_split_list:
        if split == "test": continue
        for pos in pos_ne_count[split]:
            total_pos_ne_count[pos] += pos_ne_count[split][pos]
    print "-"*80 + "\n   POS     NE   total  ratio\n" + "-"*80
    for pos, count in sorted(total_pos_ne_count.iteritems(), key=lambda x: x[1], reverse=True)[:10]:
        total = total_pos_count[pos]
        print "%6s %6d %7d %5.1f%%" % (pos, count, total, count*100./total)
    
    # Compute the mapping to labels
    ne_to_index["NONE"] = len(ne_to_index)
    
    # Add label to nodes
    for split in data_split_list:
        for tree, pyramid in data[split]["tree_pyramid_list"]:
            label_tree_data(tree, pos_to_index, ne_to_index)
            for node in pyramid:
                node.y = ne_to_index[node.ne]
    
    return (data, word_list, ne_list,
            len(character_to_index), len(pos_to_index), len(ne_to_index), len(index_to_lexicon))

if __name__ == "__main__":
    #prepare_dataset()
    #extract_glove_embeddings()
    #extract_clean_senna_lexicon()
    #extract_lexicon_embeddings()
    read_dataset()
    exit()
    
    
    
    
    
    
    
    
    
    
    
