# TF_RNN
This repository contains a special Bidirectional Recursive Neural Network implemented with Tensorflow.
```python
rnn.py # containing the RNN model class
evaluate.py # the training and testing script
ontonotes.py # utilities to extract the OntoNotes 5.0 dataset
```

## How to set up the OntoNotes 5.0 dataset
1) Get the dataset

We download it from CoNLL-2012 website.

Set its path in ontonotes.py
```python
data_path_prefix = "/home/danniel/Desktop/CONLL2012-intern/conll-2012/v4/data"
test_auto_data_path_prefix = "/home/danniel/Downloads/wu_conll_test/v9/data"
```

2) Get the load data helpers

They are provided to us by Jheng-Long Wu (jlwu@iis.sinica.edu.tw).

Import them in ontonotes.py
```python
sys.path.append("/home/danniel/Desktop/CONLL2012-intern")
from load_conll import load_data
from pstree import PSTree
```

3) Get pre-trained GloVe embeddings 

We download them from GloVe website.

Set their path in ontonotes.py
```python
glove_file = "/home/danniel/Downloads/glove.840B.300d.txt"
```

4) Extract the alphabet, vocabulary, and embeddings

Modify and run ontonotes.py
```python
if __name__ == "__main__":
    extract_vocabulary_and_alphabet()
    extract_glove_embeddings()
    # read_dataset()
    exit()
```

## How to train and test
1) Train a model pn OntoNotes.

```
python evaluate.py 2> tmp.txt
```
It generates a model file tmp.model

2) Test the model on the test split of OntoNotes

```
python evaluate.py -m evaluate -s test 2> tmp.txt
```

3) Options

To see all options, run
```
python evaluate.py -h
```
