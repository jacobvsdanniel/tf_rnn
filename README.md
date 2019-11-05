# TF_RNN
This repository contains a special Bidirectional Recursive Neural Network implemented with Tensorflow described in [1](#leveraging-linguistic-structures-for-named-entity-recognition-with-bidirectional-recursive-neural-networks).
```python
rnn.py # containing the RNN model class
evaluate.py # the training and testing script
ontonotes.py # utilities to extract the OntoNotes 5.0 dataset
```

## How to set up the OntoNotes 5.0 dataset
### 1. Get data

Download OntoNotes 5.0 from CoNLL-2012 website.

Download SENNA from Collobert's website.

Set their custom paths in ontonotes.py
```python
data_path_prefix = "/home/danniel/Desktop/CONLL2012-intern/conll-2012/v4/data"
test_auto_data_path_prefix = "/home/danniel/Downloads/wu_conll_test/v9/data"
senna_path = "/home/danniel/Downloads/senna/hash"
```

### 2. Get the load data helpers

The "load_conll_2012/" directory contains libraries to read the CoNLL-2012 format of OntoNotes. They are provided by Jheng-Long Wu (jlwu@iis.sinica.edu.tw) and Canasai (https://github.com/canasai/mps).

Set the custom path to import them in ontonotes.py
```python
sys.path.append("/home/danniel/Desktop/CONLL2012-intern")
from load_conll import load_data
from pstree import PSTree
```

### 3. Get pre-trained GloVe embeddings 

Download them from the GloVe website.

Set the custom path in ontonotes.py
```python
glove_file = "/home/danniel/Downloads/glove.840B.300d.txt"
```

### 4. Extract the alphabet, vocabulary, and embeddings

Modify and run ontonotes.py
```python
if __name__ == "__main__":
    extract_vocabulary_and_alphabet()
    extract_glove_embeddings()
    # read_dataset()
    exit()
```

## How to train and test
### 1. Train a model on OntoNotes 5.0.

```
python evaluate.py 2> tmp.txt
```
This generates model files tmp.model.*

### 2. Test the model on the test split of OntoNotes 5.0

```
python evaluate.py -m evaluate -s test 2> tmp.txt
```

### 3. Options

To see all options, run
```
python evaluate.py -h
```

## References
The high-level description of the project and the evaluation results can be found in [1](#leveraging-linguistic-structures-for-named-entity-recognition-with-bidirectional-recursive-neural-networks).

[1] PH Li, RP Dong, YS Wang, JC Chou, WY Ma, [*Leveraging Linguistic Structures for Named Entity Recognition with Bidirectional Recursive Neural Networks*](https://www.aclweb.org/anthology/D17-1282/)

```
@InProceedings{li-EtAl:2017:EMNLP20177,
  author    = {Li, Peng-Hsuan  and  Dong, Ruo-Ping  and  Wang, Yu-Siang  and  Chou, Ju-Chieh  and  Ma, Wei-Yun},
  title     = {Leveraging Linguistic Structures for Named Entity Recognition with Bidirectional Recursive Neural Networks},
  booktitle = {Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing},
  year      = {2017},
  publisher = {Association for Computational Linguistics},
  pages     = {2654--2659}
}
```
