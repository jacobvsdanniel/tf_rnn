# tf_rnn
This is the TensorFlow implementation of a recursive neural network model.
It's designed to work with data structures defined by https://github.com/ofirnachum/tree_rnn.

# Usage and testing of the model on stanford sentiment dataset

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
