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
        
        self.use_character_ngram = True
        self.alphabet_size = 5
        self.character_embeddings = 25
        self.word_length = 20
        self.max_conv_window = 3
        self.kernels = 40
        
        self.pos_dimension = 5
        self.hidden_dimension = 300
        self.output_dimension = 2
        
        self.degree = 2
        self.poses = 3
        self.words = 4
        self.neighbors = 3
        
        self.learning_rate = 1e-4
        self.epsilon = 1e-4
        self.keep_rate_P = 0.75
        self.keep_rate_X = 0.75
        self.keep_rate_H = 0.75
        self.keep_rate_R = 0.75
        return
        
class RNN(object):
    """ A Tensorflow implementation of a Recursive Neural Network
    """

    def __init__(self, config):
        self.create_hyper_parameter(config)
        self.create_input()
        self.create_word_embedding_layer()
        self.create_hidden_layer()
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
        self.e = tf.placeholder(tf.float32, [None, None])
        self.y = tf.placeholder(tf.int32, [None, None])
        self.T = tf.placeholder(tf.int32, [None, None, self.degree+3])
        self.p = tf.placeholder(tf.int32, [None, None, self.poses])
        self.x = tf.placeholder(tf.int32, [None, None, self.words])
        self.w = tf.placeholder(tf.int32, [None, None, self.words, self.word_length])
        self.S = tf.placeholder(tf.int32, [None, None, self.neighbors])
        self.l = tf.placeholder(tf.float32, [None])
        self.krP = tf.placeholder(tf.float32)
        self.krX = tf.placeholder(tf.float32)
        self.krH = tf.placeholder(tf.float32)
        self.krR = tf.placeholder(tf.float32)
        
        self.nodes = tf.shape(self.T)[0]
        self.samples = tf.shape(self.T)[1]
        
        # Create embeddings dictionaries
        with tf.variable_scope("RNN", initializer=tf.random_normal_initializer(stddev=0.1)):
            self.L = tf.get_variable("L",
                [self.vocabulary_size, self.word_to_word_embeddings])
            self.C = tf.get_variable("C",
                [2+self.alphabet_size, self.character_embeddings])
        self.L_hat = tf.concat(0, [tf.zeros([1, self.word_to_word_embeddings]), self.L])
        # self.C_hat = tf.concat(0, [tf.zeros([1, self.character_embeddings]), self.C])
        
        # Compute indices of children and neighbors
        index_offset = tf.reshape(tf.range(self.samples), [1, self.samples, 1])
        T_offset = tf.tile(index_offset, [self.nodes, 1, self.degree+3])
        self.T_hat = T_offset + (1+self.T) * self.samples
        S_offset = tf.tile(index_offset, [self.nodes, 1, self.neighbors])
        self.S_hat = S_offset + (1+self.S) * self.samples
        
        # Compute pos features
        P = tf.one_hot(self.p, self.pos_dimension, on_value=10.)
        P = tf.reshape(P, [self.nodes, self.samples, self.poses*self.pos_dimension])
        self.P = tf.nn.dropout(P, self.krP)
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
        
        X = tf.reshape(X, [self.nodes, self.samples, self.words*self.word_dimension])
        self.X = tf.nn.dropout(X, self.krX)
        
        # Mean embedding of leaf words
        m = tf.slice(self.X, [0,0,0], [self.nodes, self.samples, self.word_dimension])
        self.m = tf.reduce_sum(m, reduction_indices=0) / tf.reshape(self.l, [self.samples, 1])
        return
    """
    def get_hidden_unit(self, name):
        self.input_dimension = (self.pos_dimension * self.poses 
                              + self.word_dimension * (1+self.words))
        self.W[name] = {}
        self.b[name] = {}
        with tf.variable_scope("RNN", initializer=tf.contrib.layers.xavier_initializer()):
            with tf.variable_scope(name):
                self.W[name]["h"] = tf.get_variable("W_h",
                                        [self.input_dimension, self.hidden_dimension])
                self.b[name]["h"] = tf.get_variable("b_h", [1, self.hidden_dimension])
                
        def hidden_unit(x, c):
            h = tf.tanh(tf.matmul(x, self.W[name]["h"]) + self.b[name]["h"])
            h = tf.tanh(h + c)
            return h
        return hidden_unit
    """
    def get_hidden_unit(self, name):
        self.input_dimension = (self.pos_dimension * self.poses 
                              + self.word_dimension * (1+self.words)
                              + self.hidden_dimension)
        self.W[name] = {}
        self.b[name] = {}
        # self.a[name] = {}
        with tf.variable_scope("RNN", initializer=tf.contrib.layers.xavier_initializer()):
            with tf.variable_scope(name):
                self.W[name]["h"] = tf.get_variable("W_h",
                                        [self.input_dimension, self.hidden_dimension])
                self.b[name]["h"] = tf.get_variable("b_h", [1, self.hidden_dimension])
                # self.a[name]["h"] = tf.get_variable("a_h", initializer=0.25
                                        # * tf.ones([1, self.hidden_dimension], dtype=tf.float32))
                
        def hidden_unit(x):
            h = tf.matmul(x, self.W[name]["h"]) + self.b[name]["h"]
            
            h = tf.nn.relu(h)
            # h = tf.select(tf.greater(h, 0), h, h*self.a[name]["h"])
            return h
        return hidden_unit
    
    def create_hidden_layer(self):
        self.W = {}
        self.b = {}
        self.a = {}
        self.f_h_bottom = self.get_hidden_unit("hidden_bottom")
        self.f_h_top = self.get_hidden_unit("hidden_top")
        
        H = tf.zeros([(1+self.nodes) * self.samples, self.hidden_dimension])
        
        # Bottom-up
        def bottom_condition(index, H):
            return index <= self.nodes-1
        def bottom_body(index, H):
            p = tf.slice(self.P, [index,0,0], [1, self.samples, self.poses*self.pos_dimension])
            x = tf.slice(self.X, [index,0,0], [1, self.samples, self.words*self.word_dimension])
            
            t = tf.slice(self.T_hat, [index,0,0], [1, self.samples, self.degree])
            c = tf.reduce_sum(tf.gather(H, t[0,:,:]), reduction_indices=1)
            
            h = self.f_h_bottom(tf.concat(1, [self.m, p[0,:,:], x[0,:,:], c]))
            # h = self.f_h_bottom(tf.concat(1, [self.m, p[0,:,:], x[0,:,:]]), c)
            h = tf.nn.dropout(h, self.krH)
            
            h_upper = tf.zeros([(1+index)*self.samples, self.hidden_dimension])
            h_lower = tf.zeros([(self.nodes-1-index)*self.samples, self.hidden_dimension])
            return index+1, H+tf.concat(0, [h_upper, h, h_lower])
        _, H_bottom = tf.while_loop(bottom_condition, bottom_body, [tf.constant(0), H])
        
        # Top-down
        def top_condition(index, H):
            return index >= 0
        def top_body(index, H):
            p = tf.slice(self.P, [index,0,0], [1, self.samples, self.poses*self.pos_dimension])
            x = tf.slice(self.X, [index,0,0], [1, self.samples, self.words*self.word_dimension])
            
            t = tf.slice(self.T_hat, [index,0,self.degree+2], [1, self.samples, 1])
            c = tf.gather(H, t[0,:,0])
            
            h = self.f_h_top(tf.concat(1, [self.m, p[0,:,:], x[0,:,:], c]))
            # h = self.f_h_top(tf.concat(1, [self.m, p[0,:,:], x[0,:,:]]), c)
            h = tf.nn.dropout(h, self.krR)
            
            h_upper = tf.zeros([(1+index)*self.samples, self.hidden_dimension])
            h_lower = tf.zeros([(self.nodes-1-index)*self.samples, self.hidden_dimension])
            return index-1, H+tf.concat(0, [h_upper, h, h_lower])
        _, H_top = tf.while_loop(top_condition, top_body, [self.nodes-1, H])
        
        self.H = H_bottom + H_top
        return
    
    def get_output_unit(self, name):
        self.W[name] = {}
        self.b[name] = {}
        with tf.variable_scope("RNN", initializer=tf.contrib.layers.xavier_initializer()):
            with tf.variable_scope(name):
                self.W[name]["o"] = tf.get_variable("W_o",
                                        [self.hidden_dimension * self.neighbors,
                                         self.output_dimension])
                self.b[name]["o"] = tf.get_variable("b_o", [1, self.output_dimension])
            
        def output_unit(H):
            H = tf.gather(H, self.S_hat)
            H = tf.reshape(H, [self.nodes * self.samples, self.neighbors * self.hidden_dimension])
            O = tf.matmul(H, self.W[name]["o"]) + self.b[name]["o"]
            return O
        return output_unit
    
    def create_output(self):
        self.f_o = self.get_output_unit("output")
        
        self.O = self.f_o(self.H)
        Y_hat = tf.nn.softmax(self.O)
        self.y_hat = tf.reshape(tf.argmax(Y_hat, 1), [self.nodes, self.samples])
        
        e = tf.reshape(self.e, [self.nodes * self.samples])
        y = tf.reshape(self.y, [self.nodes * self.samples])
        Y = tf.one_hot(y, self.output_dimension, on_value=1.)
        self.loss = tf.reduce_sum(e * tf.nn.softmax_cross_entropy_with_logits(self.O, Y))
        return
    
    def create_update_op(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=self.epsilon)
        self.update_op = optimizer.minimize(self.loss)
        return
    
    def train(self, root_list):
        """ Train on a single tree
        """
        e, y, T, p, x, w, S, _, l = conll_utils.get_batch_input(root_list, self.degree)
        
        loss, _ = self.sess.run([self.loss, self.update_op],
                    feed_dict={self.e: e, self.y: y, self.T: T,
                               self.p: p, self.x: x, self.w: w,
                               self.S: S, self.l: l,
                               self.krP: self.keep_rate_P, self.krX: self.keep_rate_X,
                               self.krH: self.keep_rate_H, self.krR: self.keep_rate_R})
        return loss
    
    def evaluate(self, root_list, chunk_ne_dict_list, ne_list):
        e, y, T, p, x, w, S, chunk_list, l = conll_utils.get_batch_input(root_list, self.degree)
        
        y_hat = self.sess.run(self.y_hat,
                    feed_dict={self.T: T, 
                               self.p: p, self.x: x, self.w: w,
                               self.S: S, self.l: l,
                               self.krP: 1.0, self.krX: 1.0,
                               self.krH: 1.0, self.krR: 1.0})
        
        reals = 0.
        positives = 0.
        true_postives = 0.
        for sample in range(y.shape[1]): 
            chunk_y_dict = defaultdict(lambda: [0]*(self.output_dimension-1))
            for node, label in enumerate(y_hat[:,sample]):
                if e[node][sample] == 0: continue
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
        _, y, T, p, x, w, S, chunk_list, l = conll_utils.get_batch_input([root_node], self.degree)
        
        y_hat = self.sess.run(self.y_hat,
                    feed_dict={self.T: T, 
                               self.p: p, self.x: x, self.w: w,
                               self.S: S, self.l: l,
                               self.krP: 1.0, self.krX: 1.0,
                               self.krH: 1.0, self.krR: 1.0})
        
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

        
