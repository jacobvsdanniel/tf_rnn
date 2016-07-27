# TF_RNN
This repository contains Recursive Neural Networks implemented with Tensorflow.

# Prepare Initial Word Embeddings
1. Get glove.840B.300d.txt

2. Run
```python
import conll_utils
conll_utils.extract_glove_embeddings()
```

3. You should see glove_word.npy and glove_embedding.npy

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

# Predict Named Entities
Run
```
python conll.py
```
