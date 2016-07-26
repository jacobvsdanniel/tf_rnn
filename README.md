# TF_RNN
This repository contains Recursive Neural Networks implemented with Tensorflow.

# GloVe Word Embeddings
Word embeddings will be initialized with glove.840B.300.

In conll.py:
```python
glove_path = "."
glove_vecs = np.load(os.path.join(glove_path, "glove.npy"))
glove_words = np.load(os.path.join(glove_path, "words.npy"))
```

# CoNLL 2012 Dataset
The dataset and the imported load_conll.py was provided to me by Jheng-Long Wu (jlwu@iis.sinica.edu.tw).  

In conll_utils.py:
```python
sys.path.append("../CONLL2012-intern")
from load_conll import load_data
```

In conll.py:
```python
data_path = "../CONLL2012-intern/conll-2012/v4/data"
```

# Predicting Named Entities
Run
```
python conll.py
```
