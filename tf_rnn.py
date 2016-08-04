import numpy as np
import tensorflow as tf
import conll_utils

class Config(object):
    """ Store hyper parameters for tree models
    """
    def __init__(self, pos_dimension=5, vocabulary_size=5, embedding_dimension=300,
                 hidden_dimension=300,
                 output_dimension=2, degree=2,
                 learning_rate=0.001,
                 siblings=3, words=2):
        self.pos_dimension = pos_dimension
        self.vocabulary_size = vocabulary_size
        self.embedding_dimension = embedding_dimension
        self.hidden_dimension = hidden_dimension
        self.output_dimension = output_dimension
        self.degree = degree
        self.learning_rate = learning_rate
        self.siblings = siblings
        self.words = words
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
        self.x2 = tf.placeholder(tf.int32, [None, self.config.words])
        self.S = tf.placeholder(tf.int32, [None, self.config.siblings])
        
        with tf.variable_scope("RNN", initializer=tf.random_normal_initializer(stddev=0.1)):
            self.L = tf.get_variable("L",
                [self.config.vocabulary_size, self.config.embedding_dimension])
        
        dummy_embedding = tf.zeros([1, self.config.embedding_dimension])
        L_hat = tf.concat(0, [dummy_embedding, self.L])
        self.X1 = tf.gather(L_hat, self.x1+1)
        self.X2 = tf.gather(L_hat, self.x2+1)
        self.X2 = tf.reshape(self.X2, [-1, self.config.embedding_dimension*self.config.words])
        return
    
    def create_hidden_unit(self):
        with tf.variable_scope("RNN", initializer=tf.contrib.layers.xavier_initializer()):
            self.W_h = tf.get_variable("W_h",
                                       [self.config.hidden_dimension,
                                        self.config.embedding_dimension
                                      + self.config.hidden_dimension * self.config.degree])
            self.b_h = tf.get_variable('b_h', [self.config.hidden_dimension, 1])
        
        def hidden_unit(x1, C):
            c = tf.reshape(C, [-1,1])
            x = tf.concat(0, [x1, c])
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
        H = tf.zeros([nodes, self.config.hidden_dimension])
        
        def condition(index, H):
            return index < nodes
        
        def body(index, H):
            x1 = tf.slice(self.X1, [index,0], [1,self.config.embedding_dimension])
            x1 = tf.reshape(x1, [-1, 1])
            
            dummy_hidden = tf.zeros([1, self.config.hidden_dimension])
            H_hat = tf.concat(0, [dummy_hidden, H])
            C = tf.gather(H_hat, tf.gather(self.T, index)+1)
                
            h = self.f_h(x1, C)
            h = tf.reshape(h, [1, -1])
            
            upper = tf.zeros([index, self.config.hidden_dimension])
            lower = tf.zeros([nodes-1-index, self.config.hidden_dimension])
            H_row_hot = tf.concat(0, [upper, h, lower])
            return index+1, H+H_row_hot
        
        _, self.H = tf.while_loop(condition, body, [index, H])
        return
    
    def create_output_unit(self):
        with tf.variable_scope("RNN", initializer=tf.contrib.layers.xavier_initializer()):
            self.W_o = tf.get_variable("W_o",
                            [self.config.output_dimension,
                             self.config.hidden_dimension*self.config.siblings
                           + self.config.pos_dimension
                           + self.config.embedding_dimension*self.config.words])
            self.b_o = tf.get_variable('b_o', [self.config.output_dimension, 1])
            
        def output_unit(H):
            dummy_hidden = tf.zeros([1, self.config.hidden_dimension])
            H_hat = tf.concat(0, [dummy_hidden, H])
            H_hat = tf.gather(H_hat, self.S+1)
            H_hat = tf.reshape(H_hat, [-1, self.config.hidden_dimension*self.config.siblings])
            
            X = tf.concat(1, [H_hat, self.P, self.X2])
            O = tf.matmul(X, self.W_o, transpose_b=True) + tf.reshape(self.b_o, [-1])
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
        y1, y2, T, p, x1, x2, S = conll_utils.get_formatted_input(
                                            root_node, self.config.degree)
        Y = conll_utils.get_one_hot(y1, self.config.output_dimension)
        P = conll_utils.get_one_hot(p, self.config.pos_dimension)
        
        loss, _ = self.sess.run([self.loss, self.update_op],
                    feed_dict={self.y:Y, self.T:T, self.P:P, self.x1:x1, self.x2:x2, self.S:S})
        return loss
    
    def evaluate(self, root_node):
        y1, y2, T, p, x1, x2, S = conll_utils.get_formatted_input(
                                            root_node, self.config.degree)
        P = conll_utils.get_one_hot(p, self.config.pos_dimension)
        
        y_hat = self.sess.run(self.y_hat,
                    feed_dict={self.T:T, self.P:P, self.x1:x1, self.x2:x2, self.S:S})
        y_hat_int = np.argmax(y_hat, axis=1)
        
        correct_array = (y_hat_int == y1)
        postive_array = y_hat_int < (self.config.output_dimension - 1)
        
        true_postives = np.sum(correct_array * postive_array)
        return true_postives, np.sum(postive_array)
    
    def predict(self, root_node):
        y1, y2, T, p, x1, x2, S = conll_utils.get_formatted_input(
                                            root_node, self.config.degree)
        P = conll_utils.get_one_hot(p, self.config.pos_dimension)
        
        y_hat = self.sess.run(self.y_hat,
                    feed_dict={self.T:T, self.P:P, self.x1:x1, self.x2:x2, self.S:S})
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

        
