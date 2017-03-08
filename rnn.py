from collections import defaultdict

import numpy as np
import tensorflow as tf

class Node(object):
    def __init__(self):
        self.child_list = []
        self.parent = None
        self.left = None
        self.right = None
        
    def add_child(self, child):
        if self.child_list:
            sibling = self.child_list[-1]
            sibling.right = child
            child.left = sibling
        self.child_list.append(child)
        child.parent = self

class Config(object):
    """ Store hyper parameters for tree models
    """
    
    def __init__(self):
        self.vocabulary_size = 5
        self.word_to_word_embeddings = 300
        
        self.use_character_to_word_embedding = False
        self.alphabet_size = 5
        self.character_embeddings = 25
        self.word_length = 20
        self.max_conv_window = 3
        self.kernels = 40
        
        self.lexicons = 4
        
        self.pos_dimension = 5
        self.hidden_dimension = 450
        self.output_dimension = 2
        
        self.degree = 2
        self.poses = 3
        self.words = 4
        self.neighbors = 3
        
        self.learning_rate = 1e-4
        self.epsilon = 1e-2
        self.keep_rate_P = 0.65
        self.keep_rate_X = 0.65
        self.keep_rate_H = 0.65
        self.keep_rate_R = 0.65
        return
        
class RNN(object):
    """ A special Bidrectional Recursive Neural Network
    
    From an input tree, it classifies each node and identifies positive spans and their labels.
    
    Instantiating an object of this class only defines a Tensorflow computation graph
    under the name scope "RNN". Weights of a model instance reside in a Tensorflow session.
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
        """ Add attributes of cofig to self
        """
        for parameter in dir(config):
            if parameter[0] == "_": continue
            setattr(self, parameter, getattr(config, parameter))
        
    def create_input(self):
        """ Construct the input layer and embedding dictionaries
        
        If L is a tensor, wild indices will cause tf.gather() to raise error.
        Since L is a variable, gathering with some index of x being -1 will return zeroes,
        but will still raise error in apply_gradient.
        """
        # Create placeholders
        self.e   = tf.placeholder(tf.float32, [None, None])
        self.y   = tf.placeholder(  tf.int32, [None, None])
        self.T   = tf.placeholder(  tf.int32, [None, None, self.degree+3])
        self.p   = tf.placeholder(  tf.int32, [None, None, self.poses])
        self.x   = tf.placeholder(  tf.int32, [None, None, self.words])
        self.w   = tf.placeholder(  tf.int32, [None, None, self.words, self.word_length])
        self.lex = tf.placeholder(tf.float32, [None, None, self.lexicons])
        self.S   = tf.placeholder(  tf.int32, [None, None, self.neighbors])
        self.l   = tf.placeholder(tf.float32, [None])
        self.krP = tf.placeholder(tf.float32)
        self.krX = tf.placeholder(tf.float32)
        self.krH = tf.placeholder(tf.float32)
        self.krR = tf.placeholder(tf.float32)
        
        self.nodes = tf.shape(self.T)[0]
        self.samples = tf.shape(self.T)[1]
        
        # Create embedding dictionaries
        # We use one-hot character embeddings so no dictionary is needed
        with tf.variable_scope("RNN", initializer=tf.random_normal_initializer(stddev=0.1)):
            self.L = tf.get_variable("L",
                [self.vocabulary_size, self.word_to_word_embeddings])
            # self.C = tf.get_variable("C",
                # [2+self.alphabet_size, self.character_embeddings])
        self.L_hat = tf.concat(axis=0, values=[tf.zeros([1, self.word_to_word_embeddings]), self.L])
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
    
    def create_convolution_layers(self):
        """ Create a unit which use the character string of a word to generate its embedding
        
        Special characters: -1: start, -2: end, -3: padding
        We use one-hot character embeddings so no dictionary is needed.
        """
        self.K = [None]
        self.character_embeddings = self.alphabet_size
        with tf.variable_scope("RNN", initializer=tf.contrib.layers.xavier_initializer()):
            for window in xrange(1, self.max_conv_window+1):
                self.K.append(tf.get_variable("K%d" % window,
                                [window, self.character_embeddings, 1, self.kernels*window]))
        
        def cnn(w):
            W = tf.one_hot(w+2, self.alphabet_size, on_value=1.)
            # W = tf.gather(self.C_hat, w+3)
            W = tf.reshape(W, [-1, self.word_length, self.character_embeddings, 1])
            stride = [1, 1, self.character_embeddings, 1]
            
            W_hat = []
            for window in xrange(1, self.max_conv_window+1):
                W_window = tf.nn.conv2d(W, self.K[window], stride, "VALID")
                W_window = tf.reduce_max(W_window, axis=[1, 2])
                W_hat.append(W_window)
            
            W_hat = tf.concat(axis=1, values=W_hat)
            return tf.nn.relu(W_hat)
        
        self.f_x_cnn = cnn
        return
        
    def create_highway_layers(self):
        """ Create a unit to transform the embedding of a word from CNN 
        
        A highway layer is a linear combination of a fully connected layer and an identity layer.
        """
        layers = 1
        self.W_x_mlp = []
        self.W_x_gate = []
        self.b_x_mlp = []
        self.b_x_gate = []
        for layer in xrange(layers):
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
                
        def highway(x):
            data = x
            for layer in xrange(layers):
                mlp = tf.nn.relu(tf.matmul(data, self.W_x_mlp[layer]) + self.b_x_mlp[layer])
                gate = tf.sigmoid(tf.matmul(data, self.W_x_gate[layer]) + self.b_x_gate[layer])
                data = mlp*gate + data*(1-gate)
            return data
            
        self.f_x_highway = highway
        return
    
    def create_word_embedding_layer(self):
        """ Create a layer to compute word embeddings for all words
        """
        self.word_dimension = self.word_to_word_embeddings
        X = tf.gather(self.L_hat, self.x+1)
        
        if self.use_character_to_word_embedding:
            conv_windows = (1+self.max_conv_window) * self.max_conv_window / 2
            self.character_to_word_embeddings = conv_windows * self.kernels
            self.word_dimension += self.character_to_word_embeddings
            
            self.create_convolution_layers()
            self.create_highway_layers()
            
            w = tf.reshape(self.w, [self.nodes*self.samples*self.words, self.word_length])
            W = self.f_x_highway(self.f_x_cnn(w))
            X = tf.reshape(X, [self.nodes*self.samples*self.words, self.word_to_word_embeddings])
            X = tf.concat(axis=1, values=[X, W])
        
        X = tf.reshape(X, [self.nodes, self.samples, self.words*self.word_dimension])
        self.X = tf.nn.dropout(X, self.krX)
        
        # Mean embedding of leaf words
        m = tf.slice(self.X, [0,0,0], [self.nodes, self.samples, self.word_dimension])
        self.m = tf.reduce_sum(m, axis=0) / tf.reshape(self.l, [self.samples, 1])
        return
    
    def create_phrase_embedding_layer(self):
        """ Create a layer to compute embeddings for each phrase in lexicon
        
        Suppose there are four classes: per, org, loc, misc.
        Then "Donald Rumsfeld" will have initial embedding: [1, 0, 0, 0]
        Unsenn phrases will have dummy zero embeddings.
        """
        
        Lex = tf.gather(self.L_phrase_hat, self.lex+1)
        self.Lex = tf.reshape(Lex, [self.nodes, self.samples, self.phrases*self.phrase_embeddings])
        return
    
    def get_hidden_unit(self, name, degree):
        """ Create a unit to compute the hidden features of one direction of a node
        """
        self.input_dimension = (self.pos_dimension * self.poses 
                              + self.word_dimension * (1+self.words)
                              + self.lexicons
                              + self.hidden_dimension * degree)
        #self.input_dimension = self.lexicons
        
        self.W[name] = {}
        self.b[name] = {}
        with tf.variable_scope("RNN", initializer=tf.contrib.layers.xavier_initializer()):
            with tf.variable_scope(name):
                self.W[name]["h"] = tf.get_variable("W_h",
                                        [self.input_dimension, self.hidden_dimension])
                self.b[name]["h"] = tf.get_variable("b_h", [1, self.hidden_dimension])
                
        def hidden_unit(x):
            h = tf.matmul(x, self.W[name]["h"]) + self.b[name]["h"]
            h = tf.nn.relu(h)
            return h
        return hidden_unit
    """    
    def get_hidden_unit(self, name, degree):
        self.input_dimension = (self.pos_dimension * self.poses 
                              + self.word_dimension * (1+self.words)
                              + self.lexicons)
        
        self.W[name] = {}
        self.b[name] = {}
        with tf.variable_scope("RNN", initializer=tf.contrib.layers.xavier_initializer()):
            with tf.variable_scope(name):
                self.W[name]["z"] = tf.get_variable("W_z",
                                        [self.input_dimension, self.hidden_dimension])
                self.W[name]["f"] = tf.get_variable("W_f",
                                        [self.input_dimension, self.hidden_dimension])
                self.b[name]["z"] = tf.get_variable("b_z", [1, self.hidden_dimension])
                self.b[name]["f"] = tf.get_variable("b_f", [1, self.hidden_dimension])
                
        def hidden_unit(x, c):
            z = tf.matmul(x, self.W[name]["z"]) + self.b[name]["z"]
            z = tf.nn.relu(z)
            
            f = tf.matmul(x, self.W[name]["f"]) + self.b[name]["f"]
            f = tf.nn.sigmoid(f)
            
            h = z + tf.multiply(f,c)
            h = tf.nn.relu(h)
            return h
        return hidden_unit
    """
    def create_hidden_layer(self):
        """ Create a layer to compute hidden features for all nodes
        """
        self.W = {}
        self.b = {}
        self.a = {}
        #self.f_h_bottom = self.get_hidden_unit("hidden_bottom", self.degree)
        self.f_h_bottom = self.get_hidden_unit("hidden_bottom", 1)
        self.f_h_top = self.get_hidden_unit("hidden_top", 1)
        
        H = tf.zeros([(1+self.nodes) * self.samples, self.hidden_dimension])
        
        # Bottom-up
        def bottom_condition(index, H):
            return index <= self.nodes-1
        def bottom_body(index, H):
            p = tf.slice(self.P, [index,0,0], [1, self.samples, self.poses*self.pos_dimension])
            x = tf.slice(self.X, [index,0,0], [1, self.samples, self.words*self.word_dimension])
            lex = tf.slice(self.lex, [index,0,0], [1, self.samples, self.lexicons])
            
            t = tf.slice(self.T_hat, [index,0,0], [1, self.samples, self.degree])
            c = tf.reduce_sum(tf.gather(H, t[0,:,:]), axis=1)
            #c = tf.reshape(tf.gather(H, t[0,:,:]), [self.samples, self.degree*self.hidden_dimension])
            
            h = self.f_h_bottom(tf.concat(axis=1, values=[self.m, p[0,:,:], x[0,:,:], lex[0,:,:], c]))
            #h = self.f_h_bottom(tf.concat(axis=1, values=[self.m, p[0,:,:], x[0,:,:], lex[0,:,:]]), c)
            #h = self.f_h_bottom(lex[0,:,:])
            h = tf.nn.dropout(h, self.krH)
            
            h_upper = tf.zeros([(1+index)*self.samples, self.hidden_dimension])
            h_lower = tf.zeros([(self.nodes-1-index)*self.samples, self.hidden_dimension])
            return index+1, H+tf.concat(axis=0, values=[h_upper, h, h_lower])
        _, H_bottom = tf.while_loop(bottom_condition, bottom_body, [tf.constant(0), H])
        
        # Top-down
        def top_condition(index, H):
            return index >= 0
        def top_body(index, H):
            p = tf.slice(self.P, [index,0,0], [1, self.samples, self.poses*self.pos_dimension])
            x = tf.slice(self.X, [index,0,0], [1, self.samples, self.words*self.word_dimension])
            lex = tf.slice(self.lex, [index,0,0], [1, self.samples, self.lexicons])
            
            t = tf.slice(self.T_hat, [index,0,self.degree+2], [1, self.samples, 1])
            c = tf.gather(H, t[0,:,0])
            
            h = self.f_h_top(tf.concat(axis=1, values=[self.m, p[0,:,:], x[0,:,:], lex[0,:,:], c]))
            #h = self.f_h_top(tf.concat(axis=1, values=[self.m, p[0,:,:], x[0,:,:], lex[0,:,:]]), c)
            #h = self.f_h_top(lex[0,:,:])
            h = tf.nn.dropout(h, self.krR)
            
            h_upper = tf.zeros([(1+index)*self.samples, self.hidden_dimension])
            h_lower = tf.zeros([(self.nodes-1-index)*self.samples, self.hidden_dimension])
            return index-1, H+tf.concat(axis=0, values=[h_upper, h, h_lower])
        _, H_top = tf.while_loop(top_condition, top_body, [self.nodes-1, H])
        #_, H_top = tf.while_loop(top_condition, top_body, [self.nodes-1, H_bottom])
        
        self.H = H_bottom + H_top
        #self.H = tf.concat(axis=1, values=[H_bottom, H_top])
        #self.H = H_bottom
        #self.H = H_top
        return
    
    def get_output_unit(self, name):
        """ Create a unit to compute the class scores of a node
        """
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
        """ Construct the output layer
        """
        self.f_o = self.get_output_unit("output")
        
        self.O = self.f_o(self.H)
        Y_hat = tf.nn.softmax(self.O)
        self.y_hat = tf.reshape(tf.argmax(Y_hat, 1), [self.nodes, self.samples])
        
        e = tf.reshape(self.e, [self.nodes * self.samples])
        y = tf.reshape(self.y, [self.nodes * self.samples])
        Y = tf.one_hot(y, self.output_dimension, on_value=1.)
        self.loss = tf.reduce_sum(e * tf.nn.softmax_cross_entropy_with_logits(logits=self.O, labels=Y))
        return
    
    def create_update_op(self):
        """ Create the computation of back-propagation
        """
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=self.epsilon)
        self.update_op = optimizer.minimize(self.loss)
        return
    
    def get_padded_word(self, word):
        """ Preprocessing: Form a uniform-length string from a raw word string 
        
        Mainly to enable batch CNN.
        A word W is cut and transformed into start, W_cut, end, padding.
        Special characters: -1: start, -2: end, -3: padding.
        """
        word_cut = [-1] + word[:self.word_length-2] + [-2]
        padding = [-3] * (self.word_length - len(word_cut))
        return word_cut + padding

    def get_formatted_input(self, tree):
        """ Preprocessing: Extract data structures from an input tree
        
        The argument "tree" is actually a root node
        """
        # Get BFS layers
        layer_list = []
        layer = [tree]
        while layer:
            layer_list.append(layer)
            child_layer = []
            for node in layer:
                child_layer.extend(node.child_list)
            layer = child_layer
        
        # Index nodes bottom-up
        index = -1
        for layer in reversed(layer_list):
            for node in layer:
                index += 1
                node.index = index
        
        # Extract data from layers bottom-up
        N = []
        e = []
        y = []
        T = []
        p = []
        x = []
        w = []
        lex = []
        S = []
        l = 0
        for layer in reversed(layer_list):
            for node in layer:
                N.append(node)
                
                e.append(1)
                
                y.append(node.y)
                
                child_index_list = [child.index for child in node.child_list]
                T.append(child_index_list
                         + [-1] * (self.degree-len(node.child_list))
                         + [node.left.index if node.left else -1,
                            node.right.index if node.right else -1,
                            node.parent.index if node.parent else -1])
                
                p.append([node.pos_index,
                          node.left.pos_index if node.left else -1,
                          node.right.pos_index if node.right else -1])
                
                x.append([node.word_index,
                          node.head_index,
                          node.left.head_index if node.left else -1,
                          node.right.head_index if node.right else -1])
                
                w.append([self.get_padded_word(node.word_split),
                          self.get_padded_word(node.head_split),
                          self.get_padded_word(node.left.head_split if node.left else []),
                          self.get_padded_word(node.right.head_split if node.right else [])])
                
                lex.append(node.lexicon_hit)
                
                S.append([node.index,
                          node.left.index if node.left else -1,
                          node.right.index if node.right else -1])
                
                if node.word_index != -1: l += 1
        N   = np.array(N)
        e   = np.array(  e, dtype=np.float32)
        y   = np.array(  y, dtype=np.int32)
        T   = np.array(  T, dtype=np.int32)
        p   = np.array(  p, dtype=np.int32)
        x   = np.array(  x, dtype=np.int32)
        w   = np.array(  w, dtype=np.int32)
        lex = np.array(lex, dtype=np.float32)
        S   = np.array(  S, dtype=np.int32)
        return N, e, y, T, p, x, w, lex, S, l, tree.index
            
    def get_batch_input(self, tree_list):
        """ Preprocessing: Get batched data structures for the input layer from input trees
        """
        input_list = []
        for tree in tree_list:
            input_list.append(self.get_formatted_input(tree))
        
        samples = len(input_list)
        nodes = max([i[1].shape[0] for i in input_list])
        N   =      np.zeros([nodes, samples                              ], dtype=np.object)
        e   =      np.zeros([nodes, samples                              ], dtype=np.float32)
        y   = -1 * np.ones( [nodes, samples                              ], dtype=np.int32)
        T   = -1 * np.ones( [nodes, samples, self.degree+3               ], dtype=np.int32)
        p   = -1 * np.ones( [nodes, samples, self.poses                  ], dtype=np.int32)
        x   = -1 * np.ones( [nodes, samples, self.words                  ], dtype=np.int32)
        w   = -3 * np.ones( [nodes, samples, self.words, self.word_length], dtype=np.int32)
        lex =      np.zeros([nodes, samples, self.lexicons                ], dtype=np.float32)
        S   = -1 * np.ones( [nodes, samples, self.neighbors              ], dtype=np.int32)
        l   =      np.zeros(        samples                               , dtype=np.float32)
        r   =      np.zeros(        samples                               , dtype=np.int32)
        
        for sample, sample_input in enumerate(input_list):
            n = sample_input[0].shape[0]
            (  N[:n, sample      ],
               e[:n, sample      ],
               y[:n, sample      ],
               T[:n, sample, :   ],
               p[:n, sample, :   ],
               x[:n, sample, :   ],
               w[:n, sample, :, :],
             lex[:n, sample, :   ],
               S[:n, sample, :   ],
               l[    sample      ],
               r[    sample      ]) = sample_input
        return N, e, y, T, p, x, w, lex, S, l, r
        
    def train(self, tree_list):
        """ Update parameters from a batch of trees with labeled nodes
        """
        _, e, y, T, p, x, w, lex, S, l, _ = self.get_batch_input(tree_list)
        
        loss, _ = self.sess.run([self.loss, self.update_op],
                    feed_dict={self.e: e, self.y: y, self.T: T,
                               self.p: p, self.x: x, self.w: w, self.lex: lex,
                               self.S: S, self.l: l,
                               self.krP: self.keep_rate_P, self.krX: self.keep_rate_X,
                               self.krH: self.keep_rate_H, self.krR: self.keep_rate_R})
        return loss
    
    def predict(self, tree_list):
        """ Predict positive spans and their labels from a batch of trees
        
        Spans that are contained by other positive spans are ignored.
        """
        N, _, _, T, p, x, w, lex, S, l, r = self.get_batch_input(tree_list)
        
        y_hat = self.sess.run(self.y_hat,
                    feed_dict={self.T: T, 
                               self.p: p, self.x: x, self.w: w, self.lex: lex,
                               self.S: S, self.l: l,
                               self.krP: 1.0, self.krX: 1.0,
                               self.krH: 1.0, self.krR: 1.0})
        
        def parse_output(node_index, sample_index, span_y):
            node = N[node_index][sample_index]
            label = y_hat[node_index][sample_index]
            
            if label != self.output_dimension-1:
                span_y[node.span] = label
                return
            
            for child in node.child_list:
                parse_output(child.index, sample_index, span_y)
            return
        
        tree_span_y = []
        for sample_index in xrange(T.shape[1]): 
            span_y = {}
            parse_output(r[sample_index], sample_index, span_y)
            tree_span_y.append(span_y)
        
        return tree_span_y
       
def main():
    config = Config()
    model = RNN(config)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        L = sess.run(model.L)
        print L
        print tf.trainable_variables()
    return
    
if __name__ == "__main__":
    main()
    exit()

        
