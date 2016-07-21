import numpy as np
import tensorflow as tf
import tree_rnn

class Config(object):
    """ Store hyper parameters for tree models
    """
    def __init__(self, vocabulary_size=5, embedding_dimension=4, hidden_dimension=3,
                 output_dimension=2, degree=2,
                 learning_rate=0.001, momentum=0.9, patience=3):
        self.vocabulary_size = vocabulary_size
        self.embedding_dimension = embedding_dimension
        self.hidden_dimension = hidden_dimension
        self.output_dimension = output_dimension
        self.degree = degree
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.patience = patience
        return
        
class RNN(object):
    """ A Tensorflow implementation of a Recursive Neural Network
    
    Refer to [link to formula.pdf] for more details
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
        
        with tf.variable_scope("RNN", initializer=tf.random_normal_initializer(stddev=0.1)):
            self.L = tf.get_variable("L",
                                     [self.config.vocabulary_size,
                                      self.config.embedding_dimension])
        
        x_dummy = tf.ones_like(self.x) * self.config.vocabulary_size
        x_hat = tf.select(tf.equal(self.x, -1), x_dummy, self.x)
        L_dummy = tf.zeros([1, self.config.embedding_dimension])
        L_hat = tf.concat(0, [self.L, L_dummy])
        self.X = tf.gather(L_hat, x_hat)
        return
    
    def create_hidden_unit_concatenated(self):
        """ Create a reusable graph for computing node hidden features
        
        Childiren's vector are concatenated.
        """
        with tf.variable_scope("RNN", initializer=tf.random_normal_initializer(stddev=0.1)):
            self.W_hx = tf.get_variable("W_hx",
                                        [self.config.hidden_dimension,
                                         self.config.embedding_dimension])
            self.W_hh = tf.get_variable("W_hh",
                                        [self.config.hidden_dimension,
                                         self.config.hidden_dimension * self.config.degree])
            self.b_h = tf.get_variable('b_h', [self.config.hidden_dimension, 1])
        
        def hidden_unit(p_x, C):
            c = tf.reshape(C, [-1,1])
            p_h = tf.tanh(tf.matmul(self.W_hx,p_x) + tf.matmul(self.W_hh,c) + self.b_h)
            return p_h
        
        self.f_h = hidden_unit
        return
        
    def create_hidden_unit_summed(self):
        """ Create a reusable graph for computing node hidden features
        
        Childiren's vector are summed.
        """
        with tf.variable_scope("RNN", initializer=tf.random_normal_initializer(stddev=0.1)):
            self.W_hx = tf.get_variable("W_hx",
                                        [self.config.hidden_dimension,
                                         self.config.embedding_dimension])
            self.W_hh = tf.get_variable("W_hh",
                                        [self.config.hidden_dimension,
                                         self.config.hidden_dimension])
            self.b_h = tf.get_variable('b_h', [self.config.hidden_dimension, 1])
        
        def hidden_unit(p_x, C):
            c = tf.reshape(tf.reduce_sum(C, reduction_indices=0), [-1,1])
            p_h = tf.tanh(tf.matmul(self.W_hx,p_x) + tf.matmul(self.W_hh,c) + self.b_h)
            return p_h
        
        self.f_h = hidden_unit
        return
        
    def create_hidden_unit_first_last_sum(self):
        """ Create a reusable graph for computing node hidden features
        
        Use [fisrt c, last c, sum c] as children feature
        """
        with tf.variable_scope("RNN", initializer=tf.random_normal_initializer(stddev=0.1)):
            self.W_hx = tf.get_variable("W_hx",
                                        [self.config.hidden_dimension,
                                         self.config.embedding_dimension])
            self.W_hh = tf.get_variable("W_hh",
                                        [self.config.hidden_dimension,
                                         self.config.hidden_dimension * 3])
            self.b_h = tf.get_variable('b_h', [self.config.hidden_dimension, 1])
        
        def hidden_unit(p_x, C):
            c_sum = tf.reshape(tf.reduce_sum(C, reduction_indices=0), [1,-1])
            c_first = tf.slice(C, [0,0], [1,self.config.hidden_dimension])
            c_last = tf.slice(C, [tf.shape(C)[0]-1,0], [1,self.config.hidden_dimension])
            c = tf.reshape(tf.concat(0, [c_sum, c_first, c_last]), [-1,1])
            p_h = tf.tanh(tf.matmul(self.W_hx,p_x) + tf.matmul(self.W_hh,c) + self.b_h)
            return p_h
        
        self.f_h = hidden_unit
        return
                
    def create_recursive_hidden_function(self):
        """ Use while_loop() to construct a recursive graph
        
        Nested gather of p_x = L[x][index] in while_loop() will raise error in gradient updates.
        """
        self.create_hidden_unit_summed()
        #self.create_hidden_unit_concatenated()
        #self.create_hidden_unit_first_last_sum()
        
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
            
            def get_C():
                c_padded = tf.gather(self.T, index-leaves)
                degree = tf.reduce_sum(tf.cast(tf.not_equal(c_padded, -1), tf.int32))
                c = tf.slice(c_padded, [0], [degree])
                C = tf.gather(H, c)
                return C
            
            C = tf.cond(index < leaves,
                        lambda: tf.zeros([self.config.degree, self.config.hidden_dimension]),
                        get_C)
            
            p_h = self.f_h(p_x, C)
            p_h = tf.reshape(p_h, [1, -1])
            
            upper = tf.zeros([index, self.config.hidden_dimension])
            lower = tf.zeros([nodes-1-index, self.config.hidden_dimension])
            p_H = tf.concat(0, [upper, p_h, lower])
            return index+1, H+p_H
        
        _, self.H = tf.while_loop(condition, body, [index, H])
        return
    
    def create_output_unit(self):
        with tf.variable_scope("RNN", initializer=tf.random_normal_initializer(stddev=0.1)):
            self.W_oh = tf.get_variable("W_oh",
                                        [self.config.output_dimension,
                                         self.config.hidden_dimension])
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
        # weight = y_pos*30 + y_neg
        # self.loss = tf.reduce_sum(weight*loss)
        return
    
    def create_update_op(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
        self.update_op = optimizer.minimize(self.loss)
        return
    
    def create_update_op_backup(self):
        optimizer = tf.train.MomentumOptimizer(self.config.learning_rate, self.config.momentum)
        #self.update_op = optimizer.minimize(self.loss)
        
        g_list = optimizer.compute_gradients(self.loss)
        
        # 000
        g_list_new = [(tf.clip_by_norm(g, 5), v) for g, v in g_list]
        # g_list_new = []
        # for g, v in g_list:
            # g_not_finite = tf.logical_or(tf.is_nan(g), tf.is_inf(g))
            
            # 001
            # g = tf.select(g_not_finite, tf.zeros_like(g), g)
            # g = tf.clip_by_norm(g, 5)
            # g = tf.select(g_not_finite, 0.1*v, g)
            
            # 002
            # g = tf.convert_to_tensor(g)
            # g_norm = tf.sqrt(tf.reduce_sum(tf.square(g)))
            # g = tf.select(g_not_finite, 0.1*v, g*5/g_norm)
            
            # g_list_new.append((g, v))
        
        self.update_op = optimizer.apply_gradients(g_list_new)
        return
    
    def train(self, root_node):
        """ Train on a single tree
        
        Get integer labels from the tree and transform them to one-hot arrays.
        In (edge) case there are no internal nodes, we make sure T has rank 2.
        """
        x, T, y_int, _ = tree_rnn.gen_nn_inputs(
                            root_node,
                            max_degree = self.config.degree,
                            only_leaves_have_vals = False,
                            with_labels = True)
        samples = len(y_int)
        y = np.zeros((samples, self.config.output_dimension), dtype=np.float32)
        y[np.arange(samples), y_int.astype('int32')] = 1
        
        if T.shape[0]:
            T = T[:, :-1]
        else:
            T = np.zeros([0, self.config.degree], dtype=np.int32)
        
        loss, _ = self.sess.run([self.loss, self.update_op],
                                feed_dict={self.x:x, self.T:T, self.y:y})
        return loss
    
    def evaluate(self, root_node):
        """ Evaluate on a single tree
        
        Get integer labels from the tree and transform them to one-hot arrays.
        In (edge) case there are no internal nodes, we make sure T has rank 2.
        """
        x, T, y_int, _ = tree_rnn.gen_nn_inputs(
                            root_node,
                            max_degree = self.config.degree,
                            only_leaves_have_vals = False,
                            with_labels = True)
        
        samples = len(y_int)
        y = np.zeros((samples, self.config.output_dimension), dtype=np.float32)
        y[np.arange(samples), y_int.astype('int32')] = 1
        
        if T.shape[0]:
            T = T[:, :-1]
        else:
            T = np.zeros([0, self.config.degree], dtype=np.int32)
        
        y_hat = self.sess.run(self.y_hat, feed_dict={self.x:x, self.T:T})
        
        correct_array = (np.argmax(y_hat, axis=1) == y_int)
        postive_array = (np.argmax(y_hat, axis=1) != (self.config.output_dimension-1))
        true_postives = np.sum(correct_array * postive_array)
        precision_denominator = np.sum(postive_array)
        recall_denominator = np.sum(y_int != (self.config.output_dimension-1))
        return true_postives, precision_denominator, recall_denominator
    
    def evaluate_backup(self, root_node):
        """ Evaluate on a single tree
        
        Get integer labels from the tree and transform them to one-hot arrays.
        In (edge) case there are no internal nodes, we make sure T has rank 2.
        """
        x, T, y_int, _ = tree_rnn.gen_nn_inputs(
                            root_node,
                            max_degree = self.config.degree,
                            only_leaves_have_vals = False,
                            with_labels = True)
        
        samples = len(y_int)
        y = np.zeros((samples, self.config.output_dimension), dtype=np.float32)
        y[np.arange(samples), y_int.astype('int32')] = 1
        
        if T.shape[0]:
            T = T[:, :-1]
        else:
            T = np.zeros([0, self.config.degree], dtype=np.int32)
        
        y_hat = self.sess.run(self.y_hat, feed_dict={self.x:x, self.T:T})
        
        corrects = np.sum(np.argmax(y_hat, axis=1) == y_int)
        return corrects, samples
    
    def predict(self, root_node):
        x, T = tree_rnn.gen_nn_inputs(root_node, max_degree=self.config.degree,
                                      only_leaves_have_vals=False)
        y_hat = self.sess.run(self.y_hat, feed_dict={self.x:x, self.T:T[:, :-1]})
        return y_hat

    def train_backup(self, root_node, y):
        x, T = tree_rnn.gen_nn_inputs(root_node, max_degree=self.config.degree,
                                      only_leaves_have_vals=False)
        
        loss, y_hat, _ = self.sess.run([self.loss, self.y_hat, self.update_op],
                                       feed_dict={self.x:x, self.T:T[:, :-1], self.y:y})
        return loss, y_hat
    
    def train_step(self, root_node, not_used):
        """ Compatible to train_step() in sentiment.py
        
        Get integer labels from tree and transform them to one-hot arrays.
        """
        x, T, y_int, _ = tree_rnn.gen_nn_inputs(root_node,
                                                max_degree=self.config.degree,
                                                only_leaves_have_vals=False,
                                                with_labels=True)
        samples = len(y_int)
        y = np.zeros((samples, self.config.output_dimension), dtype=np.float32)
        y[np.arange(samples), y_int.astype('int32')] = 1
        
        loss, y_hat, _ = self.sess.run([self.loss, self.y_hat, self.update_op],
                                       feed_dict={self.x:x, self.T:T[:, :-1], self.y:y})
        return loss, y_hat
    
def main():
    config = Config()
    model = RNN(config)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        L = sess.run(model.L)
        print L
    return
    
if __name__ == "__main__":
    main()
    exit()
        
        
