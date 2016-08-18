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
        
        self.alphabet_size = 5
        self.word_length = 20
        self.character_dimension = 25
        self.max_conv_window = 5
        self.kernels = 40
        self.embedding_dimension_2 = sum(xrange(1, self.max_conv_window+1)) * self.kernels
        
        self.hidden_dimension = 300
        self.output_dimension = 2
        
        self.degree = 2
        self.node_features = 3
        
        self.learning_rate = 1e-4
        self.epsilon = 1e-4
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
        self.y = tf.placeholder(tf.int32, [None, None])
        self.T = tf.placeholder(tf.int32, [None, None, self.degree])
        self.p = tf.placeholder(tf.int32, [None, None])
        
        self.x1 = tf.placeholder(tf.int32, [None, None])
        self.x2 = tf.placeholder(tf.int32, [None, None])
        self.x3 = tf.placeholder(tf.int32, [None, None])
        self.xx = tf.placeholder(tf.int32, [None, None, self.word_length])
        
        self.S = tf.placeholder(tf.int32, [None, None, self.node_features])
        
        with tf.variable_scope("RNN", initializer=tf.random_normal_initializer(stddev=0.1)):
            self.L = tf.get_variable("L",
                [self.vocabulary_size, self.embedding_dimension])
            self.C = tf.get_variable("C",
                [self.alphabet_size+2, self.character_dimension])
        
        dummy_embedding = tf.zeros([1, self.embedding_dimension])
        L_hat = tf.concat(0, [dummy_embedding, self.L])
        self.X1 = tf.gather(L_hat, self.x1+1)
        self.X2 = tf.gather(L_hat, self.x2+1)
        self.X3 = tf.gather(L_hat, self.x3+1)
        
        self.P = tf.one_hot(self.p, self.pos_dimension, on_value=5.)
        
        dummy_character = tf.zeros([1, self.character_dimension])
        self.C_hat = tf.concat(0, [dummy_character, self.C])
        
        self.samples = tf.shape(self.x1)[0]
        self.nodes = tf.shape(self.x1)[1]
        return
    
    def create_word_cnn_unit(self):
        self.K = [None]
        with tf.variable_scope("RNN", initializer=tf.contrib.layers.xavier_initializer()):
            for window in xrange(1, self.max_conv_window+1):
                self.K.append(tf.get_variable("K%d" % window,
                                [window, self.character_dimension, 1, self.kernels*window]))
        
        def word_cnn_unit(xx):
            XX = tf.gather(self.C_hat, xx+3)
            XX = tf.reshape(XX, [self.samples * self.nodes * 3,
                                 self.word_length, self.character_dimension, 1])
            stride = [1, 1, self.character_dimension, 1]
            
            XX_hat = []
            for window in xrange(1, self.max_conv_window+1):
                XX_window = tf.nn.conv2d(XX, self.K[window], stride, "VALID")
                XX_window = tf.reduce_max(XX_window, reduction_indices=[1, 2])
                XX_hat.append(XX_window)
            
            XX_hat = tf.concat(1, XX_hat)
            return tf.tanh(XX_hat)
        
        self.f_x_cnn = word_cnn_unit
        return
        
    def create_word_highway_unit(self):
        layers = 1
        self.W_x_mlp = []
        self.W_x_gate = []
        self.b_x_mlp = []
        self.b_x_gate = []
        for layer in range(layers):
            with tf.variable_scope("RNN", initializer=tf.contrib.layers.xavier_initializer()):
                self.W_x_mlp.append(tf.get_variable("W_x_mlp_%d" % layer,
                                            [self.embedding_dimension_2,
                                             self.embedding_dimension_2]))
                self.W_x_gate.append(tf.get_variable("W_x_gate_%d" % layer,
                                            [self.embedding_dimension_2,
                                             self.embedding_dimension_2]))
                self.b_x_mlp.append(tf.get_variable("b_x_mlp_%d" % layer,
                                            [1, self.embedding_dimension_2]))
            with tf.variable_scope("RNN",
                        initializer=tf.random_normal_initializer(mean=-2, stddev=0.1)):
                self.b_x_gate.append(tf.get_variable("b_x_gate_%d" % layer,
                                            [1, self.embedding_dimension_2]))
                
        def word_highway_unit(xx):
            data = xx
            for layer in range(layers):
                mlp = tf.tanh(tf.matmul(data, self.W_x_mlp[layer]) + self.b_x_mlp[layer])
                gate = tf.sigmoid(tf.matmul(data, self.W_x_gate[layer]) + self.b_x_gate[layer])
                data = mlp*gate + data*(1-gate)
            return data
        self.f_x_highway = word_highway_unit
        return
        
    def create_character_embedding_layer(self):
        self.create_word_cnn_unit()
        self.create_word_highway_unit()
        
        xx = tf.reshape(self.xx, [self.samples*self.nodes*3, self.word_length])
        XX = self.f_x_highway(self.f_x_cnn(xx))
        self.XX = tf.reshape(XX, [self.samples, self.nodes*3, self.embedding_dimension_2])
        
    def create_hidden_unit(self):
        with tf.variable_scope("RNN", initializer=tf.contrib.layers.xavier_initializer()):
            self.W_h = tf.get_variable("W_h",
                                       [self.pos_dimension
                                        + self.embedding_dimension * 3
                                        + self.embedding_dimension_2 * 3
                                        + self.hidden_dimension,
                                        self.hidden_dimension])
            self.b_h = tf.get_variable('b_h', [1, self.hidden_dimension])
        
        def hidden_unit(x):
            h = tf.matmul(x, self.W_h) + self.b_h
            return tf.tanh(h)
            # return tf.nn.elu(h)
        
        self.f_h = hidden_unit
        return
    
    def create_recursive_hidden_layer(self):
        """ Use while_loop() to construct a recursive graph
        
        Nested gather of x = L[x][index] in while_loop() will raise error in gradient updates.
        """
        self.create_hidden_unit()
        
        index = tf.constant(0)
        H = tf.zeros([self.samples, self.nodes, self.hidden_dimension])
        
        t_offset = tf.reshape(tf.range(self.samples)*(1+self.nodes), [self.samples, 1])
        t_offset = tf.tile(t_offset, [1, self.degree])
        
        def condition(index, H):
            return index < self.nodes
        
        def body(index, H):
            p = tf.slice(self.P, [0, index, 0], [self.samples, 1, self.pos_dimension])
            x1 = tf.slice(self.X1, [0, index, 0], [self.samples, 1, self.embedding_dimension])
            x2 = tf.slice(self.X2, [0, index, 0], [self.samples, 1, self.embedding_dimension])
            x3 = tf.slice(self.X3, [0, index, 0], [self.samples, 1, self.embedding_dimension])
            xx = tf.slice(self.XX, [0, index*3, 0], [self.samples, 3, self.embedding_dimension_2])
            
            p = tf.reshape(p, [self.samples, self.pos_dimension])
            x1 = tf.reshape(x1, [self.samples, self.embedding_dimension])
            x2 = tf.reshape(x2, [self.samples, self.embedding_dimension])
            x3 = tf.reshape(x3, [self.samples, self.embedding_dimension])
            xx = tf.reshape(xx, [self.samples, self.embedding_dimension_2*3])
            
            dummy_hidden = tf.zeros([self.samples, 1, self.hidden_dimension])
            H_hat = tf.concat(1, [dummy_hidden, H])
            H_hat = tf.reshape(H_hat, [self.samples*(1+self.nodes), self.hidden_dimension])
            t = tf.slice(self.T, [0, index, 0], [self.samples, 1, self.degree])
            t = t_offset + tf.reshape(t, [self.samples, self.degree]) + 1
            C = tf.gather(H_hat, t)
            c = tf.reduce_sum(C, reduction_indices=1)
            
            h = self.f_h(tf.concat(1, [p, x1, x2, x3, xx, c]))
            # h = self.f_h(tf.concat(1, [p, x1, x2, x3, c]))
            H_upper = tf.zeros([self.samples, index, self.hidden_dimension])
            H_h = tf.reshape(h, [self.samples, 1, self.hidden_dimension])
            H_lower = tf.zeros([self.samples, self.nodes-1-index, self.hidden_dimension])
            H_row_hot = tf.concat(1, [H_upper, H_h, H_lower])
            return index+1, H+H_row_hot
        
        _, self.H = tf.while_loop(condition, body, [index, H])
        return
    
    def create_output_unit(self):
        with tf.variable_scope("RNN", initializer=tf.contrib.layers.xavier_initializer()):
            self.W_o = tf.get_variable("W_o",
                            [self.node_features * self.hidden_dimension,
                             self.output_dimension])
            self.b_o = tf.get_variable('b_o', [1, self.output_dimension])
            
        def output_unit(H, S):
            S_offset = tf.reshape(tf.range(self.samples)*(1+self.nodes), [self.samples, 1])
            S_offset = tf.tile(S_offset, [1, self.nodes*self.node_features])
            S_hat = S_offset + tf.reshape(S, [self.samples, self.nodes*self.node_features]) + 1 
            
            dummy_hidden = tf.zeros([self.samples, 1, self.hidden_dimension])
            H_hat = tf.concat(1, [dummy_hidden, H])
            H_hat = tf.reshape(H_hat, [self.samples*(1+self.nodes), self.hidden_dimension])
            H_hat = tf.gather(H_hat, S_hat)
            H_hat = tf.reshape(H_hat, [self.samples * self.nodes,
                                       self.node_features * self.hidden_dimension])
            
            O = tf.matmul(H_hat, self.W_o) + self.b_o
            # O = tf.reshape(H_hat, [self.samples, self.nodes, self.output_dimension])
            return O
        
        self.f_o = output_unit
        return
        
    def create_output(self):
        self.create_output_unit()
        
        self.O = self.f_o(self.H, self.S)
        Y_hat = tf.nn.softmax(self.O)
        self.y_hat = tf.reshape(tf.argmax(Y_hat, 1), [self.samples, self.nodes])
        
        y = tf.reshape(self.y, [self.samples * self.nodes])
        Y = tf.one_hot(y, self.output_dimension, on_value=1.)
        self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(self.O, Y))
        # self.loss = loss / tf.cast(self.samples, dtype=tf.float32)
        return
    
    def create_update_op(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=self.epsilon)
        self.update_op = optimizer.minimize(self.loss)
        return
    
    def train(self, root_list):
        """ Train on a single tree
        """
        (y1, T, p, x1, x2, x3, xx, S, chunk_list
            ) = conll_utils.get_batch_input(root_list, self.degree)
        
        loss, _ = self.sess.run([self.loss, self.update_op],
                    feed_dict={self.y: y1, self.T: T, self.p: p,
                                self.x1: x1, self.x2: x2, self.x3: x3, self.xx: xx,
                                self.S:S})
        return loss
    
    def evaluate(self, root_list, chunk_ne_dict_list, ne_list):
        (y1, T, p, x1, x2, x3, xx, S, chunk_list
            ) = conll_utils.get_batch_input(root_list, self.degree)
        
        y_hat = self.sess.run(self.y_hat,
                    feed_dict={self.T: T, self.p: p,
                                self.x1: x1, self.x2: x2, self.x3: x3, self.xx: xx,
                                self.S:S})
        
        reals = 0.
        positives = 0.
        true_postives = 0.
        for sample in range(len(y1)): 
            chunk_y_dict = defaultdict(lambda: [0]*(self.output_dimension-1))
            for node, y in enumerate(y_hat[sample]):
                if y1[sample][node] == -1: continue
                if y == self.output_dimension - 1: continue
                chunk_y_dict[chunk_list[sample][node]][y] += 1
                
            reals += len(chunk_ne_dict_list[sample])
            positives += len(chunk_y_dict)
            for chunk in chunk_y_dict.iterkeys():
                if chunk not in chunk_ne_dict_list[sample]: continue
                if chunk_ne_dict_list[sample][chunk] == ne_list[np.argmax(chunk_y_dict[chunk])]:
                    true_postives += 1
        
        return true_postives, positives, reals
    
    def predict(self, root_node):
        (y1, y2, T, p, x1, x2, x3, xx, S, chunk_list
            ) = conll_utils.get_formatted_input(root_node, self.degree)
        
        y_hat = self.sess.run(self.y_hat,
                    feed_dict={self.T: T, self.p: p,
                                self.x1: x1, self.x2: x2, self.x3: x3, self.xx: xx,
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

        
