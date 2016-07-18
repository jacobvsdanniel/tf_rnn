# tf_rnn
This is the TensorFlow implementation of a recursive neural network model.
It's designed to work with data structures defined by https://github.com/ofirnachum/tree_rnn.

# Example changes to test this model in sentiment.py
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
```python
# Get word embeddings data structure from model
embeddings = model.sess.run(model.L)
# codes that fill in embeddings #
## ... ##
# Update embeddings to model
update = model.L.assign(embeddings)
model.sess.run(update)
```
Result should be
```
82% accuracy on coarse-grained sentiment data
40% accuracy on find-grained sentiment data
```
