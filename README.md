# tf_rnn
This is the TensorFlow implementation of a recursive neural network model.
It's designed to work with data structures defined by https://github.com/ofirnachum/tree_rnn.

# Test on CoNLL2012 (ontonotes 5.0), targeting parser labels

Run
```
python conll.py
```

In conll_utils.py, I imported from load_conll.py a function that load raw data from CoNLL2012. It was originally main.py provided by Jheng-Long Wu (jlwu@iis.sinica.edu.tw).  
```python
sys.path.append("../CONLL2012-intern")
from load_conll import load_data
```

Dataset
```
training sentences
  75187
validation sentences
  9603
Test sentences
  9479
total samples (nodes in parse trees)
  3118378
classes
  79
```

Expected result
```
91% accuracy
```

# Test on Stanford sentiment dataset

Edit sentiment.py
```python
# Initialize a model:
config = tf_rnn.Config(
    embedding_dimension=EMB_DIM,
    vocabulary_size=num_emb, 
    hidden_dimension=HIDDEN_DIM,
    output_dimension=output_dim,
    degree=max_degree,
    learning_rate=LEARNING_RATE)
model = tf_rnn.RNN(config)
model.sess = tf.Session()
model.sess.run(tf.initialize_all_variables())
```

More editing...
```python
# Get word embeddings data structure from model
embeddings = model.sess.run(model.L)
# codes that fill in embeddings #
## ... ##
# Update embeddings to model
update = model.L.assign(embeddings)
model.sess.run(update)
```

Expected Results
```
Each epoch should take 2 to 3 minutes.
Coarse-grained sentiment data
    82% accuracy (concatenated children)
Find-grained sentiment data
    41% accuracy (concatenated children)
    43% accuracy (summed children)
```
