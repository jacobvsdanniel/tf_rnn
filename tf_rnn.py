from collections import defaultdict

import numpy as np
import tensorflow as tf

import conll_utils

class Config(object):
    """ Store hyper parameters for tree models
    """
    def __init__(self):
        self.pos_dimension = 5
        self.vocabulary_size = 5
        self.embedding_dimension = 300
        self.hidden_dimension = 300
        self.output_dimension = 2
        self.degree = 2
        self.siblings = 3
        self.word_features = 2
        self.pos_features = 1
        self.learning_rate = 1e-3
        self.epsilon = 1e-1
        return
        
class RNN(object):
    """ A Tensorflow implementation of a Recursive Neural Network
    """

    def __init__(self, config):
        self.create_hyper_parameter(config)
        self.create_input()
        self.create_recursive_hidden_function()
        self.create_output()
        self.create_update_op()
        return
        
    def create_hyper_parameter(self, config):
        for parameter in dir(config):
            if parameter[0] == "_": continue
            setattr(self, parameter, getattr(config, parameter))
        
    def create_input(self):
        """ Prepare input embeddings
        
        If L is a tensor, wild indices will cause tf.gather() to raise error.
        Since L is a variable, gathering with some index of x being -1 will return zeroes,
        but will still raise error in apply_gradient.
        """
        self.y = tf.placeholder(tf.float32, [None, self.output_dimension])
        self.T = tf.placeholder(tf.int32, [None, self.degree])
        self.p = tf.placeholder(tf.int32, [None, self.pos_features])
        self.x1 = tf.placeholder(tf.int32, [None])
        self.x2 = tf.placeholder(tf.int32, [None, self.word_features])
        self.S = tf.placeholder(tf.int32, [None, self.siblings])
        
        with tf.variable_scope("RNN", initializer=tf.random_normal_initializer(stddev=0.1)):
            self.L = tf.get_variable("L",
                [self.vocabulary_size, self.embedding_dimension])
        
        dummy_embedding = tf.zeros([1, self.embedding_dimension])
        L_hat = tf.concat(0, [dummy_embedding, self.L])
        self.X1 = tf.gather(L_hat, self.x1+1)
        self.X2 = tf.gather(L_hat, self.x2+1)
        self.X2 = tf.reshape(self.X2, [-1, self.embedding_dimension*self.word_features])
        
        dummy_pos = np.zeros([1, self.pos_dimension], dtype=np.float32)
        P = np.identity(self.pos_dimension, dtype=np.float32)
        P = tf.constant(np.concatenate((dummy_pos, P)))
        self.P = tf.gather(P, self.p+1)
        self.P = tf.reshape(self.P, [-1, self.pos_dimension*self.pos_features])
        return
    
    def create_hidden_unit(self):
        with tf.variable_scope("RNN", initializer=tf.contrib.layers.xavier_initializer()):
            self.W_h = tf.get_variable("W_h",
                                       [self.hidden_dimension,
                                        self.embedding_dimension
                                      + self.hidden_dimension
                                      + self.embedding_dimension*self.word_features
                                      + self.pos_dimension*self.pos_features])
            self.b_h = tf.get_variable('b_h', [self.hidden_dimension, 1])
        
        def hidden_unit(x1, C, x2, p):
            c = tf.reshape(tf.reduce_sum(C, reduction_indices=0), [-1,1])
            x = tf.concat(0, [x1, c, x2, p])
            h = tf.tanh(tf.matmul(self.W_h, x) + self.b_h)
            return h
        
        self.f_h = hidden_unit
        return
    
    def create_recursive_hidden_function(self):
        """ Use while_loop() to construct a recursive graph
        
        Nested gather of x = L[x][index] in while_loop() will raise error in gradient updates.
        """
        self.create_hidden_unit()
        
        nodes = tf.shape(self.x1)[0]
        index = tf.constant(0)
        H = tf.zeros([nodes, self.hidden_dimension])
        
        def condition(index, H):
            return index < nodes
        
        def body(index, H):
            x1 = tf.slice(self.X1, [index,0], [1,self.embedding_dimension])
            x1 = tf.reshape(x1, [-1, 1])
            
            dummy_hidden = tf.zeros([1, self.hidden_dimension])
            H_hat = tf.concat(0, [dummy_hidden, H])
            C = tf.gather(H_hat, tf.gather(self.T, index)+1)
            
            x2 = tf.slice(self.X2, [index,0], [1,self.embedding_dimension*self.word_features])
            x2 = tf.reshape(x2, [-1, 1])
            
            p = tf.slice(self.P, [index,0], [1,self.pos_dimension*self.pos_features])
            p = tf.reshape(p, [-1, 1])
            
            h = self.f_h(x1, C, x2, p)
            h = tf.reshape(h, [1, -1])
            
            upper = tf.zeros([index, self.hidden_dimension])
            lower = tf.zeros([nodes-1-index, self.hidden_dimension])
            H_row_hot = tf.concat(0, [upper, h, lower])
            return index+1, H+H_row_hot
        
        _, self.H = tf.while_loop(condition, body, [index, H])
        return
    
    def create_output_unit(self):
        with tf.variable_scope("RNN", initializer=tf.contrib.layers.xavier_initializer()):
            self.W_o = tf.get_variable("W_o",
                            [self.output_dimension,
                             self.hidden_dimension*self.siblings])
            self.b_o = tf.get_variable('b_o', [self.output_dimension, 1])
            
        def output_unit(H):
            dummy_hidden = tf.zeros([1, self.hidden_dimension])
            H_hat = tf.concat(0, [dummy_hidden, H])
            H_hat = tf.gather(H_hat, self.S+1)
            H_hat = tf.reshape(H_hat, [-1, self.hidden_dimension*self.siblings])
            # X = tf.concat(1, [H_hat, self.X2, self.P])
            O = tf.matmul(H_hat, self.W_o, transpose_b=True) + tf.reshape(self.b_o, [-1])
            return O
        
        self.f_o = output_unit
        return
        
    def create_output(self):
        self.create_output_unit()
        
        self.O = self.f_o(self.H)
        self.y_hat = tf.nn.softmax(self.O)
        
        self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(self.O, self.y))
        return
    
    def create_update_op(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=self.epsilon)
        self.update_op = optimizer.minimize(self.loss)
        return
    
    def train(self, root_node):
        """ Train on a single tree
        
        Get integer labels from the tree and transform them to one-hot arrays.
        In (edge) case there are no internal nodes, we make sure T has rank 2.
        """
        y1, y2, T, p, x1, x2, S, chunk_list = conll_utils.get_formatted_input(
                                                root_node, self.degree)
        
        Y = conll_utils.get_one_hot(y1, self.output_dimension)
        
        loss, _ = self.sess.run([self.loss, self.update_op],
                    feed_dict={self.y:Y, self.T:T, self.p:p, self.x1:x1, self.x2:x2, self.S:S})
        return loss
    
    def evaluate(self, root_node, chunk_ne_dict, ne_list):
        y1, y2, T, p, x1, x2, S, chunk_list = conll_utils.get_formatted_input(
                                                root_node, self.degree)
        
        y_hat = self.sess.run(self.y_hat,
                    feed_dict={self.T:T, self.p:p, self.x1:x1, self.x2:x2, self.S:S})
        y_hat_int = np.argmax(y_hat, axis=1)
        
        chunk_y_dict = defaultdict(lambda: [0]*(self.output_dimension-1))
        for index, y in enumerate(y_hat_int):
            if y == self.output_dimension - 1: continue
            chunk_y_dict[chunk_list[index]][y] += 1
            
        reals = len(chunk_ne_dict)
        positives = len(chunk_y_dict)
        true_postives = 0
        for chunk in chunk_y_dict.iterkeys():
            if chunk not in chunk_ne_dict: continue
            if chunk_ne_dict[chunk] == ne_list[np.argmax(chunk_y_dict[chunk])]:
                true_postives += 1
                
        return true_postives, positives, reals
    
    def predict(self, root_node):
        y1, y2, T, p, x1, x2, S, chunk_list = conll_utils.get_formatted_input(
                                                root_node, self.degree)
        
        y_hat = self.sess.run(self.y_hat,
                    feed_dict={self.T:T, self.p:p, self.x1:x1, self.x2:x2, self.S:S})
        y_hat_int = np.argmax(y_hat, axis=1)
        
        confusion_matrix = np.zeros([19, 19], dtype=np.int32)
        for node in range(y1.shape[0]):
            confusion_matrix[y1[node]][y_hat_int[node]] += 1
        return confusion_matrix
        
def main():
    config = Config()
    model = RNN(config)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        L = sess.run(model.L)
        print L
        print tf.trainable_variables()
    return
    
if __name__ == "__main__":
    main()
    exit()

        
