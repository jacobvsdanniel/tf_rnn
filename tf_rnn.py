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
        self.x = tf.placeholder(tf.int32, [None])
        self.T = tf.placeholder(tf.int32, [None, self.config.degree])
        self.y = tf.placeholder(tf.float32, [None, self.config.output_dimension])
        self.P = tf.placeholder(tf.float32, [None, self.config.pos_dimension])
        self.a = tf.placeholder(tf.int32, [None])
        
        with tf.variable_scope("RNN", initializer=tf.random_normal_initializer(stddev=0.1)):
            self.L = tf.get_variable("L",
                [self.config.vocabulary_size, self.config.embedding_dimension])
        
        dummy_embedding = tf.zeros([1, self.config.embedding_dimension])
        L_hat = tf.concat(0, [self.L, dummy_embedding])
        dummy_index = tf.ones_like(self.x) * self.config.vocabulary_size
        
        x_hat = tf.select(tf.equal(self.x, -1), dummy_index, self.x)
        self.X = tf.gather(L_hat, x_hat)
        
        a_hat = tf.select(tf.equal(self.a, -1), dummy_index, self.a)
        self.A = tf.gather(L_hat, a_hat)
        return
    
    def create_hidden_unit(self):
        with tf.variable_scope("RNN", initializer=tf.contrib.layers.xavier_initializer()):
            self.W_hp = tf.get_variable("W_hp",
                                        [self.config.hidden_dimension,
                                         self.config.pos_dimension])
            self.W_hx = tf.get_variable("W_hx",
                                        [self.config.hidden_dimension,
                                         self.config.embedding_dimension])
            self.W_ha = tf.get_variable("W_ha",
                                        [self.config.hidden_dimension,
                                         self.config.embedding_dimension])
            self.W_hh = tf.get_variable("W_hh",
                                        [self.config.hidden_dimension,
                                         self.config.hidden_dimension])
            self.b_h = tf.get_variable('b_h', [self.config.hidden_dimension, 1])
        
        def hidden_unit(p_x, C, p_p, p_a):
            c = tf.reshape(tf.reduce_sum(C, reduction_indices=0), [-1,1])
            p_h = tf.tanh(tf.matmul(self.W_hx, p_x)
                        + tf.matmul(self.W_hh, c)
                        + tf.matmul(self.W_hp, p_p)
                        + tf.matmul(self.W_ha, p_a)
                        + self.b_h)
            return p_h
        
        self.f_h = hidden_unit
        return
    
    def create_recursive_hidden_function(self):
        """ Use while_loop() to construct a recursive graph
        
        Nested gather of p_x = L[x][index] in while_loop() will raise error in gradient updates.
        """
        self.create_hidden_unit()
        
        nodes = tf.shape(self.x)[0]
        leaves = nodes - tf.shape(self.T)[0]
        
        index = tf.constant(0)
        H = tf.zeros([nodes, self.config.hidden_dimension])
        
        def condition(index, H):
            return index < nodes
        
        def body(index, H):
            #p_x = tf.gather(self.X, index)
            p_x = tf.slice(self.X, [index,0], [1,self.config.embedding_dimension])
            p_x = tf.reshape(p_x, [-1, 1])
            
            p_p = tf.slice(self.P, [index,0], [1,self.config.pos_dimension])
            p_p = tf.reshape(p_p, [-1, 1])
            
            p_a = tf.slice(self.A, [index,0], [1,self.config.embedding_dimension])
            p_a = tf.reshape(p_a, [-1, 1])
            
            def get_C():
                c_padded = tf.gather(self.T, index-leaves)
                degree = tf.reduce_sum(tf.cast(tf.not_equal(c_padded, -1), tf.int32))
                c = tf.slice(c_padded, [0], [degree])
                C = tf.gather(H, c)
                return C
            
            C = tf.cond(index < leaves,
                        lambda: tf.zeros([self.config.degree, self.config.hidden_dimension]),
                        get_C)
            
            p_h = self.f_h(p_x, C, p_p, p_a)
            p_h = tf.reshape(p_h, [1, -1])
            
            upper = tf.zeros([index, self.config.hidden_dimension])
            lower = tf.zeros([nodes-1-index, self.config.hidden_dimension])
            p_H = tf.concat(0, [upper, p_h, lower])
            return index+1, H+p_H
        
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
        
        # Weighted loss
        # loss = tf.nn.softmax_cross_entropy_with_logits(self.O, self.y)
        # y_int = tf.argmax(self.y, dimension=1)
        # y_pos = tf.cast(tf.not_equal(y_int, self.config.output_dimension-1), dtype=tf.float32)
        # y_neg = tf.cast(tf.equal(y_int, self.config.output_dimension-1), dtype=tf.float32)
        # weight = y_pos + y_neg*0
        # self.loss = tf.reduce_sum(weight*loss)

        # Mask the loss of DONT_CARE nodes
        # loss = tf.nn.softmax_cross_entropy_with_logits(self.O, self.y)
        # y_int = tf.argmax(self.y, dimension=1)
        # care = tf.cast(tf.not_equal(y_int, self.config.output_dimension-1), dtype=tf.float32)
        # self.loss = tf.reduce_sum(loss * care)
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
        x, T, y_int, p, a = conll_utils.get_formatted_input(root_node, self.config.degree)
        
        if T.shape[0]:
            T = T[:, :-1]
        else:
            T = np.zeros([0, self.config.degree], dtype=np.int32)
        
        samples = len(y_int)
        y = np.zeros((samples, self.config.output_dimension), dtype=np.float32)
        y[np.arange(samples), y_int] = 1
        
        P = np.zeros((samples, self.config.pos_dimension), dtype=np.float32)
        P[np.arange(samples), p] = 5
        
        loss, _ = self.sess.run([self.loss, self.update_op],
                                feed_dict={self.x:x, self.T:T, self.y:y, self.P:P, self.a:a})
        return loss
    
    def evaluate(self, root_node):
        """ Evaluate on a single tree
        
        Get integer labels from the tree and transform them to one-hot arrays.
        In (edge) case there are no internal nodes, we make sure T has rank 2.
        """
        x, T, y_int, p, a = conll_utils.get_formatted_input(root_node, self.config.degree)
                            
        if T.shape[0]:
            T = T[:, :-1]
        else:
            T = np.zeros([0, self.config.degree], dtype=np.int32)
        
        samples = len(p)
        P = np.zeros((samples, self.config.pos_dimension), dtype=np.float32)
        P[np.arange(samples), p] = 5
        
        y_hat = self.sess.run(self.y_hat, feed_dict={self.x:x, self.T:T, self.P:P, self.a:a})
        y_hat_int = np.argmax(y_hat, axis=1)
        correct_array = (y_hat_int/self.config.pos_dimension == y_int/self.config.pos_dimension)
        
        # last 79 labels are pos-NONE
        postive_array = y_hat_int < (self.config.output_dimension - self.config.pos_dimension)
        
        true_postives = np.sum(correct_array * postive_array)
        return true_postives, np.sum(postive_array)
        
    def predict(self, root_node):
        x, T, y_int, p = conll_utils.get_formatted_input(root_node, self.config.degree)
                            
        if T.shape[0]:
            T = T[:, :-1]
        else:
            T = np.zeros([0, self.config.degree], dtype=np.int32)
        
        samples = len(p)
        P = np.zeros((samples, self.config.pos_dimension), dtype=np.float32)
        P[np.arange(samples), p] = 5
        
        y_hat = self.sess.run(self.y_hat, feed_dict={self.x:x, self.T:T, self.P:P})
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
        
        
