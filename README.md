# TF_RNN
This repository contains a special Bidirectional Recursive Neural Network implemented with Tensorflow.
```python
rnn.py # containing the RNN model class
evaluate.py # the training and testing script
ontonotes.py # utilities to extract the OntoNotes 5.0 dataset
```

## How to set up the OntoNotes 5.0 dataset
### 1. Get data

Download OntoNotes 5.0 from CoNLL-2012 website.

Download SENNA from Collobert's website.

Set their paths in ontonotes.py
```python
data_path_prefix = "/home/danniel/Desktop/CONLL2012-intern/conll-2012/v4/data"
test_auto_data_path_prefix = "/home/danniel/Downloads/wu_conll_test/v9/data"
senna_path = "/home/danniel/Downloads/senna/hash"
```

### 2. Get the load data helpers

The "external/" directory contains libraries that read the CoNLL-2012 format of OntoNotes (Provided by Jheng-Long Wu, jlwu@iis.sinica.edu.tw). They in turn use libraries from https://github.com/canasai/mps. Download missing files from that repository  when import errors occur.

Set custom path and import them in ontonotes.py
```python
sys.path.append("/home/danniel/Desktop/CONLL2012-intern")
from load_conll import load_data
from pstree import PSTree
```

### 3. Get pre-trained GloVe embeddings 

Download them from the GloVe website.

Set custom path in ontonotes.py
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
