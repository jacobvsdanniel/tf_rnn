from collections import defaultdict

import numpy as np
import tensorflow as tf

import conll_utils

class Config(object):
    """ Store hyper parameters for tree models
    """
    def __init__(self):
        self.pos_dimension = 5
        
        self.alphabet_size = 5
        self.character_dimension_1 = 25
        self.character_dimension_2 = 20
        self.max_pools = 1
        self.embedding_dimension_2 = sum(xrange(1, self.max_pools+1)) * self.character_dimension_2
        
        self.vocabulary_size = 5
        self.embedding_dimension = 300
        
        self.hidden_dimension = 300
        self.output_dimension = 2
        
        self.degree = 2
        self.node_features = 3
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
        self.create_character_embedding_layer()
        self.create_recursive_hidden_layer()
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
        self.x2 = tf.placeholder(tf.int32, [None])
        self.x3 = tf.placeholder(tf.int32, [None])
        self.s1 = tf.placeholder(tf.int32, [None])
        self.s2 = tf.placeholder(tf.int32, [None])
        self.s3 = tf.placeholder(tf.int32, [None])
        self.cut1 = tf.placeholder(tf.int32, [None, 2])
        self.cut2 = tf.placeholder(tf.int32, [None, 2])
        self.cut3 = tf.placeholder(tf.int32, [None, 2])
        
        self.S = tf.placeholder(tf.int32, [None, self.node_features])
        
        with tf.variable_scope("RNN", initializer=tf.random_normal_initializer(stddev=0.1)):
            self.L = tf.get_variable("L",
                [self.vocabulary_size, self.embedding_dimension])
            self.C = tf.get_variable("C",
                [self.alphabet_size+2, self.character_dimension_1])
        
        dummy_embedding = tf.zeros([1, self.embedding_dimension])
        L_hat = tf.concat(0, [dummy_embedding, self.L])
        self.X1 = tf.gather(L_hat, self.x1+1)
        self.X2 = tf.gather(L_hat, self.x2+1)
        self.X3 = tf.gather(L_hat, self.x3+1)
        
        dummy_pos = np.zeros([1, self.pos_dimension], dtype=np.float32)
        P = np.identity(self.pos_dimension, dtype=np.float32)
        P = tf.constant(np.concatenate((dummy_pos, P)))
        self.P = tf.gather(P, self.p+1)
        self.P = tf.reshape(self.P, [-1, self.pos_dimension*self.pos_features])
        return
    
    def create_character_unit(self):
        with tf.variable_scope("RNN", initializer=tf.contrib.layers.xavier_initializer()):
            self.W_c = tf.get_variable("W_c",
                                       [self.character_dimension_1*3, self.character_dimension_2])
            self.b_c = tf.get_variable('b_c', [1, self.character_dimension_2])
        
        def character_unit(C1):
            c2 = tf.matmul(C1, self.W_c) + self.b_c
            return tf.tanh(c2)
            
        self.f_c = character_unit
        return
    
    def create_word_unit(self):
        self.create_character_unit()
        
        def word_unit(w):
            characters = tf.shape(w)[0]
            w_left = tf.concat(0, [tf.constant([-1]), tf.slice(w, [0], [characters-1])])
            w_right = tf.concat(0, [tf.slice(w, [1], [characters-1]), tf.constant([-2])])
            w_hat = tf.transpose(tf.pack([w_left, w, w_right]))
            
            # Convolve windowed image with shape(W)[0] kernels
            W = tf.gather(self.C, w_hat+2)
            W = tf.reshape(W, [-1, self.character_dimension_1*3])
            W_hat = self.f_c(W)
            
            # Pool with output lengths from 1 ~ (e.g.) 3
            # P = tf.zeros([0, self.character_dimension_2])
            # for pools in range(1, self.max_pools+1):
                # remains = tf.mod(characters, pools)
                # padding_shape = [[0,pools-remains], [0,0]]
                # W_padded = tf.cond(tf.equal(remains, 0),
                                    # lambda: W_hat,
                                    # lambda: tf.pad(W_hat, padding_shape))
                # W_split = tf.split(0, pools, W_padded)
                # p = tf.reduce_max(W_split, reduction_indices=1)
                # P = tf.concat(0, [P, p])
            # W_hat = tf.reshape(P, [-1])
            
            # Pool with output lengths 1
            W_hat = tf.reduce_max(W_hat, reduction_indices=0)
            return W_hat
            
        self.f_w = word_unit
        return
    
    def create_sentence_unit(self):
        self.create_word_unit()
        
        def sentence_unit(s, cut):
            def body(cut):
                w = tf.slice(s, [cut[0]], [cut[1]])
                w_hat = tf.cond(tf.equal(cut[1],0),
                                lambda: tf.zeros([self.embedding_dimension_2]),
                                lambda: self.f_w(w))
                return w_hat
            s_hat = tf.map_fn(body, cut, dtype=tf.float32)
            return s_hat
            
        self.f_s = sentence_unit
        return
    
    def create_character_embedding_layer(self):
        self.create_sentence_unit()
        self.S1 = self.f_s(self.s1, self.cut1)
        self.S2 = self.f_s(self.s2, self.cut2)
        self.S3 = self.f_s(self.s3, self.cut3)
    
    def create_hidden_unit(self):
        with tf.variable_scope("RNN", initializer=tf.contrib.layers.xavier_initializer()):
            self.W_h = tf.get_variable("W_h",
                                       [self.pos_dimension * self.pos_features
                                        + self.embedding_dimension * 3
                                        + self.embedding_dimension_2 * 3
                                        + self.hidden_dimension,
                                        self.hidden_dimension])
            self.b_h = tf.get_variable('b_h', [1, self.hidden_dimension])
        
        def hidden_unit(x):
            h = tf.matmul(x, self.W_h) + self.b_h
            return tf.tanh(h)
        
        self.f_h = hidden_unit
        return
    
    def create_recursive_hidden_layer(self):
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
            p = tf.slice(self.P, [index,0], [1,self.pos_dimension*self.pos_features])
            x1 = tf.slice(self.X1, [index,0], [1,self.embedding_dimension])
            x2 = tf.slice(self.X2, [index,0], [1,self.embedding_dimension])
            x3 = tf.slice(self.X3, [index,0], [1,self.embedding_dimension])
            s1 = tf.slice(self.S1, [index,0], [1,self.embedding_dimension_2])
            s2 = tf.slice(self.S2, [index,0], [1,self.embedding_dimension_2])
            s3 = tf.slice(self.S3, [index,0], [1,self.embedding_dimension_2])
            
            dummy_hidden = tf.zeros([1, self.hidden_dimension])
            H_hat = tf.concat(0, [dummy_hidden, H])
            C = tf.gather(H_hat, tf.gather(self.T, index)+1)
            c = tf.reduce_sum(C, reduction_indices=0)
            
            h = self.f_h(tf.concat(1, [p, x1, x2, x3, s1, s2, s3, [c]]))
            upper = tf.zeros([index, self.hidden_dimension])
            lower = tf.zeros([nodes-1-index, self.hidden_dimension])
            H_row_hot = tf.concat(0, [upper, h, lower])
            return index+1, H+H_row_hot
        
        _, self.H = tf.while_loop(condition, body, [index, H])
        return
    
    def create_output_unit(self):
        with tf.variable_scope("RNN", initializer=tf.contrib.layers.xavier_initializer()):
            self.W_o = tf.get_variable("W_o",
                            [self.hidden_dimension*self.node_features,
                                self.output_dimension])
            self.b_o = tf.get_variable('b_o', [1, self.output_dimension])
            
        def output_unit(H, S):
            dummy_hidden = tf.zeros([1, self.hidden_dimension])
            H_hat = tf.concat(0, [dummy_hidden, H])
            H_hat = tf.gather(H_hat, S+1)
            H_hat = tf.reshape(H_hat, [-1, self.hidden_dimension*self.node_features])
            O = tf.matmul(H_hat, self.W_o) + self.b_o
            return O
        
        self.f_o = output_unit
        return
        
    def create_output(self):
        self.create_output_unit()
        
        self.O = self.f_o(self.H, self.S)
        self.y_hat = tf.nn.softmax(self.O)
        
        self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(self.O, self.y))
        return
    
    def create_update_op(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=self.epsilon)
        self.update_op = optimizer.minimize(self.loss)
        return
    
    def train(self, root_node):
        """ Train on a single tree
        """
        (y1, y2, T, p, x1, x2, x3, s1, s2, s3, cut1, cut2, cut3, S, chunk_list
            ) = conll_utils.get_formatted_input(root_node, self.degree)
        
        Y = conll_utils.get_one_hot(y1, self.output_dimension)
        
        loss, _ = self.sess.run([self.loss, self.update_op],
                    feed_dict={self.y: Y, self.T: T, self.p: p,
                                self.x1: x1, self.x2: x2, self.x3: x3,
                                self.s1: s1, self.s2: s2, self.s3: s3,
                                self.cut1: cut1, self.cut2: cut2, self.cut3: cut3,
                                self.S:S})
        return loss
    
    def evaluate(self, root_node, chunk_ne_dict, ne_list):
        (y1, y2, T, p, x1, x2, x3, s1, s2, s3, cut1, cut2, cut3, S, chunk_list
            ) = conll_utils.get_formatted_input(root_node, self.degree)
        
        y_hat = self.sess.run(self.y_hat,
                    feed_dict={self.T: T, self.p: p,
                                self.x1: x1, self.x2: x2, self.x3: x3,
                                self.s1: s1, self.s2: s2, self.s3: s3,
                                self.cut1: cut1, self.cut2: cut2, self.cut3: cut3,
                                self.S:S})
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
        (y1, y2, T, p, x1, x2, x3, s1, s2, s3, cut1, cut2, cut3, S, chunk_list
            ) = conll_utils.get_formatted_input(root_node, self.degree)
        
        y_hat = self.sess.run(self.y_hat,
                    feed_dict={self.T: T, self.p: p,
                                self.x1: x1, self.x2: x2, self.x3: x3,
                                self.s1: s1, self.s2: s2, self.s3: s3,
                                self.cut1: cut1, self.cut2: cut2, self.cut3: cut3,
                                self.S:S})
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

        
