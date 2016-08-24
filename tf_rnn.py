from collections import defaultdict

import numpy as np
import tensorflow as tf

import conll_utils

class Config(object):
    """ Store hyper parameters for tree models
    """
    def __init__(self):
        self.vocabulary_size = 5
        self.word_to_word_embeddings = 300
        
        self.use_character_ngram = False
        self.alphabet_size = 5
        self.character_embeddings = 25
        self.word_length = 20
        self.max_conv_window = 3
        self.kernels = 40
        
        self.pos_dimension = 5
        self.hidden_dimension = 300
        self.output_dimension = 2
        
        self.degree = 2
        self.poses = 6
        self.words = 7
        self.neighbors = 3
        
        self.learning_rate = 1e-4
        self.epsilon = 1e-4
        self.keep_rate = 0.8
        return
        
class RNN(object):
    """ A Tensorflow implementation of a Recursive Neural Network
    """

    def __init__(self, config):
        self.create_hyper_parameter(config)
        self.create_input()
        self.create_word_embedding_layer()
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
        # Create placeholders
        self.y = tf.placeholder(tf.int32, [None, None])
        self.T = tf.placeholder(tf.int32, [None, None, self.degree])
        self.p = tf.placeholder(tf.int32, [None, None, self.poses])
        self.x = tf.placeholder(tf.int32, [None, None, self.words])
        self.w = tf.placeholder(tf.int32, [None, None, self.words, self.word_length])
        self.S = tf.placeholder(tf.int32, [None, None, self.neighbors])
        self.kr = tf.placeholder(tf.float32)
        
        self.nodes = tf.shape(self.T)[0]
        self.samples = tf.shape(self.T)[1]
        
        # Create embeddings dictionaries
        with tf.variable_scope("RNN", initializer=tf.random_normal_initializer(stddev=0.1)):
            self.L = tf.get_variable("L",
                [self.vocabulary_size, self.word_to_word_embeddings])
            # self.C = tf.get_variable("C",
                # [2+self.alphabet_size, self.character_embeddings])
        self.L_hat = tf.concat(0, [tf.zeros([1, self.word_to_word_embeddings]), self.L])
        # self.C_hat = tf.concat(0, [tf.zeros([1, self.character_embeddings]), self.C])
        
        # Compute indices of children and neighbors
        index_offset = tf.reshape(tf.range(self.samples), [1, self.samples, 1])
        T_offset = tf.tile(index_offset, [self.nodes, 1, self.degree])
        self.T_hat = T_offset + (1+self.T) * self.samples
        S_offset = tf.tile(index_offset, [self.nodes, 1, self.neighbors])
        self.S_hat = S_offset + (1+self.S) * self.samples
        
        # Compute pos features
        P = tf.one_hot(self.p, self.pos_dimension, on_value=10.)
        self.P = tf.reshape(P, [self.nodes, self.samples, self.poses*self.pos_dimension])
        return
    
    def create_character_ngram_unit(self):
        self.K = [None]
        self.character_embeddings = self.alphabet_size
        with tf.variable_scope("RNN", initializer=tf.contrib.layers.xavier_initializer()):
            for window in xrange(1, self.max_conv_window+1):
                self.K.append(tf.get_variable("K%d" % window,
                                [window, self.character_embeddings, 1, self.kernels*window]))
        
        def character_ngram_unit(w):
            W = tf.one_hot(w+2, self.alphabet_size, on_value=1.)
            # W = tf.gather(self.C_hat, w+3)
            W = tf.reshape(W, [-1, self.word_length, self.character_embeddings, 1])
            stride = [1, 1, self.character_embeddings, 1]
            
            W_hat = []
            for window in xrange(1, self.max_conv_window+1):
                W_window = tf.nn.conv2d(W, self.K[window], stride, "VALID")
                W_window = tf.reduce_max(W_window, reduction_indices=[1, 2])
                W_hat.append(W_window)
            
            W_hat = tf.concat(1, W_hat)
            return tf.nn.relu(W_hat)
        
        self.f_x_cnn = character_ngram_unit
        return
        
    def create_word_highway_unit(self):
        layers = 0
        self.W_x_mlp = []
        self.W_x_gate = []
        self.b_x_mlp = []
        self.b_x_gate = []
        for layer in range(layers):
            with tf.variable_scope("RNN", initializer=tf.contrib.layers.xavier_initializer()):
                self.W_x_mlp.append(tf.get_variable("W_x_mlp_%d" % layer,
                                            [self.character_to_word_embeddings,
                                             self.character_to_word_embeddings]))
                self.W_x_gate.append(tf.get_variable("W_x_gate_%d" % layer,
                                            [self.character_to_word_embeddings,
                                             self.character_to_word_embeddings]))
                self.b_x_mlp.append(tf.get_variable("b_x_mlp_%d" % layer,
                                            [1, self.character_to_word_embeddings]))
            with tf.variable_scope("RNN",
                        initializer=tf.random_normal_initializer(mean=-2, stddev=0.1)):
                self.b_x_gate.append(tf.get_variable("b_x_gate_%d" % layer,
                                            [1, self.character_to_word_embeddings]))
                
        def word_highway_unit(x):
            data = x
            for layer in range(layers):
                mlp = tf.nn.relu(tf.matmul(data, self.W_x_mlp[layer]) + self.b_x_mlp[layer])
                gate = tf.sigmoid(tf.matmul(data, self.W_x_gate[layer]) + self.b_x_gate[layer])
                data = mlp*gate + data*(1-gate)
            return data
        self.f_x_highway = word_highway_unit
        return
    
    def create_word_embedding_layer(self):
        self.word_dimension = self.word_to_word_embeddings
        X = tf.gather(self.L_hat, self.x+1)
        
        if self.use_character_ngram:
            conv_windows = (1+self.max_conv_window) * self.max_conv_window / 2
            self.character_to_word_embeddings = conv_windows * self.kernels
            self.word_dimension += self.character_to_word_embeddings
            
            self.create_character_ngram_unit()
            self.create_word_highway_unit()
            
            w = tf.reshape(self.w, [self.nodes*self.samples*self.words, self.word_length])
            W = self.f_x_highway(self.f_x_cnn(w))
            X = tf.reshape(X, [self.nodes*self.samples*self.words, self.word_to_word_embeddings])
            X = tf.concat(1, [X, W])
        
        self.X = tf.reshape(X, [self.nodes, self.samples, self.words*self.word_dimension])
        return
    
    def create_hidden_unit(self):
        with tf.variable_scope("RNN", initializer=tf.contrib.layers.xavier_initializer()):
            self.W_h = tf.get_variable("W_h",
                                         [self.pos_dimension * self.poses
                                          + self.word_dimension * self.words
                                          + self.hidden_dimension,
                                          self.hidden_dimension])
            self.b_h = tf.get_variable("b_h", [1, self.hidden_dimension])
        
        def hidden_unit(x):
            # h = tf.matmul(x, tf.nn.dropout(self.W_h, self.kr)) + self.b_h
            h = tf.matmul(tf.nn.dropout(x, self.kr), self.W_h) + self.b_h
            # h = tf.matmul(x, self.W_h) + self.b_h
            # return tf.tanh(h)
            # return tf.nn.elu(h)
            return tf.nn.relu(h)
        
        self.f_h = hidden_unit
        return
    
    def create_recursive_hidden_layer(self):
        """ Use while_loop() to construct a recursive graph
        
        Nested gather of x = L[x][index] in while_loop() will raise error in gradient updates.
        """
        self.create_hidden_unit()
        
        # self.P = tf.nn.dropout(self.P, self.kr)
        # self.X = tf.nn.dropout(self.X, self.kr)
        
        index = tf.constant(0)
        H = tf.zeros([(1+self.nodes) * self.samples, self.hidden_dimension])
        
        def condition(index, H):
            return index < self.nodes
        
        def body(index, H):
            p = tf.slice(self.P, [index,0,0], [1, self.samples, self.poses*self.pos_dimension])
            x = tf.slice(self.X, [index,0,0], [1, self.samples, self.words*self.word_dimension])
            
            t = tf.slice(self.T_hat, [index,0,0], [1, self.samples, self.degree])
            c = tf.reduce_sum(tf.gather(H, t[0,:,:]), reduction_indices=1)
            
            h = self.f_h(tf.concat(1, [p[0,:,:], x[0,:,:], c]))
            h_upper = tf.zeros([(1+index)*self.samples, self.hidden_dimension])
            h_lower = tf.zeros([(self.nodes-1-index)*self.samples, self.hidden_dimension])
            return index+1, H+tf.concat(0, [h_upper, h, h_lower])
        
        # _, H = tf.while_loop(condition, body, [index, H])
        # self.H = tf.nn.dropout(H, self.kr)
        _, self.H = tf.while_loop(condition, body, [index, H])
        return
    
    def create_output_unit(self):
        with tf.variable_scope("RNN", initializer=tf.contrib.layers.xavier_initializer()):
            self.W_o = tf.get_variable("W_o",
                            [self.hidden_dimension * self.neighbors,
                             self.output_dimension])
            self.b_o = tf.get_variable("b_o", [1, self.output_dimension])
            
        def output_unit(H):
            H_hat = tf.gather(H, self.S_hat)
            H_hat = tf.reshape(H_hat, [self.nodes * self.samples,
                                       self.neighbors * self.hidden_dimension])
            O = tf.matmul(H_hat, self.W_o) + self.b_o
            # O = tf.matmul(H_hat, tf.nn.dropout(self.W_o, self.kr)) + self.b_o
            return O
        
        self.f_o = output_unit
        return
        
    def create_output(self):
        self.create_output_unit()
        
        self.O = self.f_o(self.H)
        Y_hat = tf.nn.softmax(self.O)
        self.y_hat = tf.reshape(tf.argmax(Y_hat, 1), [self.nodes, self.samples])
        
        y = tf.reshape(self.y, [self.nodes * self.samples])
        Y = tf.one_hot(y, self.output_dimension, on_value=1.)
        self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(self.O, Y))
        return
    
    def create_update_op(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=self.epsilon)
        self.update_op = optimizer.minimize(self.loss)
        return
    
    def train(self, root_list):
        """ Train on a single tree
        """
        y, T, p, x, w, S, _ = conll_utils.get_batch_input(root_list, self.degree)
        
        loss, _ = self.sess.run([self.loss, self.update_op],
                    feed_dict={self.y: y, self.T: T,
                               self.p: p, self.x: x, self.w: w,
                               self.S:S, self.kr: self.keep_rate})
        return loss
    
    def evaluate(self, root_list, chunk_ne_dict_list, ne_list):
        y, T, p, x, w, S, chunk_list = conll_utils.get_batch_input(root_list, self.degree)
        
        y_hat = self.sess.run(self.y_hat,
                    feed_dict={self.T: T, 
                               self.p: p, self.x: x, self.w: w,
                               self.S:S, self.kr: 1.0})
        
        reals = 0.
        positives = 0.
        true_postives = 0.
        for sample in range(y.shape[1]): 
            chunk_y_dict = defaultdict(lambda: [0]*(self.output_dimension-1))
            for node, label in enumerate(y_hat[:,sample]):
                if y[node][sample] == -1: continue
                if label == self.output_dimension - 1: continue
                chunk_y_dict[chunk_list[sample][node]][label] += 1
                
            reals += len(chunk_ne_dict_list[sample])
            positives += len(chunk_y_dict)
            for chunk in chunk_y_dict.iterkeys():
                if chunk not in chunk_ne_dict_list[sample]: continue
                if chunk_ne_dict_list[sample][chunk] == ne_list[np.argmax(chunk_y_dict[chunk])]:
                    true_postives += 1
        
        return true_postives, positives, reals
    
    def predict(self, root_node):
        y, T, p, x, w, S, chunk_list = conll_utils.get_batch_input([root_node], self.degree)
        
        y_hat = self.sess.run(self.y_hat,
                    feed_dict={self.T: T, 
                               self.p: p, self.x: x, self.w: w,
                               self.S:S, self.kr: 1.0})
        
        y, y_hat = y[:,0], y_hat[:,0]
        confusion_matrix = np.zeros([19, 19], dtype=np.int32)
        for node in range(y.shape[0]):
            confusion_matrix[y[node]][y_hat[node]] += 1
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

        
