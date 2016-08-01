import numpy as np
import tensorflow as tf
import conll_utils

class Config(object):
    """ Store hyper parameters for tree models
    """
    def __init__(self, pos_dimension=5, vocabulary_size=5, embedding_dimension=300,
                 hidden_dimension=300,
                 output_dimension=2, degree=2,
                 learning_rate=0.001):
        self.pos_dimension = pos_dimension
        self.vocabulary_size = vocabulary_size
        self.embedding_dimension = embedding_dimension
        self.hidden_dimension = hidden_dimension
        self.output_dimension = output_dimension
        self.degree = degree
        self.learning_rate = learning_rate
        return
        
class RNN(object):
    """ A Tensorflow implementation of a Recursive Neural Network
    """

    def __init__(self, config):
        self.config = config
        self.create_input()
        self.create_recursive_hidden_function()
        self.create_output()
        self.create_update_op()
        return
        
    def create_input(self):
        """ Prepare input embeddings
        
        If L is a tensor, wild indices will cause tf.gather() to raise error.
        Since L is a variable, gathering with some index of x being -1 will return zeroes,
        but will still raise error in apply_gradient.
        """
        self.y = tf.placeholder(tf.float32, [None, self.config.output_dimension])
        self.T = tf.placeholder(tf.int32, [None, self.config.degree])
        self.P = tf.placeholder(tf.float32, [None, self.config.pos_dimension])
        self.x1 = tf.placeholder(tf.int32, [None])
        self.x2 = tf.placeholder(tf.int32, [None])
        self.x3 = tf.placeholder(tf.int32, [None])
        
        with tf.variable_scope("RNN", initializer=tf.random_normal_initializer(stddev=0.1)):
            self.L = tf.get_variable("L",
                [self.config.vocabulary_size, self.config.embedding_dimension])
        
        dummy_embedding = tf.zeros([1, self.config.embedding_dimension])
        L_hat = tf.concat(0, [self.L, dummy_embedding])
        dummy_index = tf.ones_like(self.x1) * self.config.vocabulary_size
        
        x1_hat = tf.select(tf.equal(self.x1, -1), dummy_index, self.x1)
        self.X1 = tf.gather(L_hat, x1_hat)
        
        x2_hat = tf.select(tf.equal(self.x2, -1), dummy_index, self.x2)
        self.X2 = tf.gather(L_hat, x2_hat)
        
        x3_hat = tf.select(tf.equal(self.x3, -1), dummy_index, self.x3)
        self.X3 = tf.gather(L_hat, x3_hat)
        return
    
    def create_hidden_unit(self):
        with tf.variable_scope("RNN", initializer=tf.contrib.layers.xavier_initializer()):
            self.W_hp = tf.get_variable("W_hp",
                                        [self.config.hidden_dimension,
                                         self.config.pos_dimension])
            self.W_hx1 = tf.get_variable("W_hx1",
                                        [self.config.hidden_dimension,
                                         self.config.embedding_dimension])
            self.W_hx2 = tf.get_variable("W_hx2",
                                        [self.config.hidden_dimension,
                                         self.config.embedding_dimension])
            self.W_hx3 = tf.get_variable("W_hx3",
                                        [self.config.hidden_dimension,
                                         self.config.embedding_dimension])
            self.W_hh = tf.get_variable("W_hh",
                                        [self.config.hidden_dimension,
                                         self.config.hidden_dimension])
            self.b_h = tf.get_variable('b_h', [self.config.hidden_dimension, 1])
        
        def hidden_unit(p, x1, x2, x3, C):
            c = tf.reshape(tf.reduce_sum(C, reduction_indices=0), [-1,1])
            h = tf.tanh(tf.matmul(self.W_hp, p)
                      + tf.matmul(self.W_hx1, x1)
                      + tf.matmul(self.W_hx2, x2)
                      + tf.matmul(self.W_hx3, x3)
                      + tf.matmul(self.W_hh, c)
                      + self.b_h)
            return h
        
        self.f_h = hidden_unit
        return
    
    def create_recursive_hidden_function(self):
        """ Use while_loop() to construct a recursive graph
        
        Nested gather of x = L[x][index] in while_loop() will raise error in gradient updates.
        """
        self.create_hidden_unit()
        
        nodes = tf.shape(self.x1)[0]
        leaves = nodes - tf.shape(self.T)[0]
        
        index = tf.constant(0)
        H = tf.zeros([nodes, self.config.hidden_dimension])
        
        def condition(index, H):
            return index < nodes
        
        def body(index, H):
            p = tf.slice(self.P, [index,0], [1,self.config.pos_dimension])
            p = tf.reshape(p, [-1, 1])
            
            #p_x = tf.gather(self.X, index)
            x1 = tf.slice(self.X1, [index,0], [1,self.config.embedding_dimension])
            x1 = tf.reshape(x1, [-1, 1])
            
            x2 = tf.slice(self.X2, [index,0], [1,self.config.embedding_dimension])
            x2 = tf.reshape(x2, [-1, 1])
            
            x3 = tf.slice(self.X3, [index,0], [1,self.config.embedding_dimension])
            x3 = tf.reshape(x3, [-1, 1])
            
            def get_C():
                c_padded = tf.gather(self.T, index-leaves)
                degree = tf.reduce_sum(tf.cast(tf.not_equal(c_padded, -1), tf.int32))
                c = tf.slice(c_padded, [0], [degree])
                C = tf.gather(H, c)
                return C
            C = tf.cond(index < leaves,
                        lambda: tf.zeros([self.config.degree, self.config.hidden_dimension]),
                        get_C)
            
            h = self.f_h(p, x1, x2, x3, C)
            h = tf.reshape(h, [1, -1])
            
            upper = tf.zeros([index, self.config.hidden_dimension])
            lower = tf.zeros([nodes-1-index, self.config.hidden_dimension])
            H_row_hot = tf.concat(0, [upper, h, lower])
            return index+1, H+H_row_hot
        
        _, self.H = tf.while_loop(condition, body, [index, H])
        return
    
    def create_output_unit(self):
        with tf.variable_scope("RNN", initializer=tf.contrib.layers.xavier_initializer()):
            self.W_oh = tf.get_variable("W_oh",
                            [self.config.output_dimension, self.config.hidden_dimension])
            self.b_o = tf.get_variable('b_o', [self.config.output_dimension, 1])
        
        def output_unit(H):
            O = tf.matmul(H, self.W_oh, transpose_b=True) + tf.reshape(self.b_o, [-1])
            #O = tf.clip_by_value(O, 0, 20)
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
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
        self.update_op = optimizer.minimize(self.loss)
        return
    
    def train(self, root_node):
        """ Train on a single tree
        
        Get integer labels from the tree and transform them to one-hot arrays.
        In (edge) case there are no internal nodes, we make sure T has rank 2.
        """
        y_int, T, p, x1, x2, x3 = conll_utils.get_formatted_input(root_node, self.config.degree)
        
        if T.shape[0]:
            T = T[:, :-1]
        else:
            T = np.zeros([0, self.config.degree], dtype=np.int32)
        
        samples = len(y_int)
        y = np.zeros((samples, self.config.output_dimension), dtype=np.float32)
        y[np.arange(samples), y_int] = 1
        
        P = np.zeros((samples, self.config.pos_dimension), dtype=np.float32)
        P[np.arange(samples), p] = 5
        P = P * np.not_equal(p,-1).reshape((samples,1))
        
        loss, _ = self.sess.run([self.loss, self.update_op],
                    feed_dict={self.y:y, self.T:T, self.P:P, self.x1:x1, self.x2:x2, self.x3:x3})
        return loss
    
    def evaluate(self, root_node):
        y_int, T, p, x1, x2, x3 = conll_utils.get_formatted_input(root_node, self.config.degree)
                            
        if T.shape[0]:
            T = T[:, :-1]
        else:
            T = np.zeros([0, self.config.degree], dtype=np.int32)
        
        samples = len(p)
        P = np.zeros((samples, self.config.pos_dimension), dtype=np.float32)
        P[np.arange(samples), p] = 5
        P = P * np.not_equal(p,-1).reshape((samples,1))
        
        y_hat = self.sess.run(self.y_hat,
                    feed_dict={self.T:T, self.P:P, self.x1:x1, self.x2:x2, self.x3:x3})
        y_hat_int = np.argmax(y_hat, axis=1)
        correct_array = (y_hat_int/self.config.pos_dimension == y_int/self.config.pos_dimension)
        
        # last pos_dimension labels are pos-NONE
        postive_array = y_hat_int < (self.config.output_dimension - self.config.pos_dimension)
        
        true_postives = np.sum(correct_array * postive_array)
        return true_postives, np.sum(postive_array)
    
    def predict(self, root_node):
        y_int, T, p, x1, x2, x3 = conll_utils.get_formatted_input(root_node, self.config.degree)
                            
        if T.shape[0]:
            T = T[:, :-1]
        else:
            T = np.zeros([0, self.config.degree], dtype=np.int32)
        
        samples = len(p)
        P = np.zeros((samples, self.config.pos_dimension), dtype=np.float32)
        P[np.arange(samples), p] = 5
        P = P * np.not_equal(p,-1).reshape((samples,1))
        
        y_hat = self.sess.run(self.y_hat,
                    feed_dict={self.T:T, self.P:P, self.x1:x1, self.x2:x2, self.x3:x3})
        y_hat_int = np.argmax(y_hat, axis=1) / self.config.pos_dimension
        y_int = y_int / self.config.pos_dimension
        
        confusion_matrix = np.zeros([19, 19], dtype=np.int32)
        for node in range(y_int.shape[0]):
            confusion_matrix[y_int[node]][y_hat_int[node]] += 1
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

        
