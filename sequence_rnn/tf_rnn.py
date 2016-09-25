from collections import defaultdict

import numpy as np
import tensorflow as tf

import conll_utils
cnt = 0

class Config(object):
    """ Store hyper parameters for tree models
    """
    def __init__(self):
        self.vocabulary_size = 5
        self.word_to_word_embeddings = 300
        
        self.pos_dimension = 5
        self.hidden_dimension = 300
        self.output_dimension = 2
        
        self.poses = 1
        self.words = 1
        
        self.learning_rate = 1e-4
        self.epsilon = 1e-4
        self.keep_rate_P = .8
        self.keep_rate_X = .8
        self.keep_rate_H = .8
        self.keep_rate_R = .8
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
        self.y = tf.placeholder(tf.int32, [None, None])
        self.p = tf.placeholder(tf.int32, [None, None, self.poses])
        self.x = tf.placeholder(tf.int32, [None, None, self.words])
        self.e = tf.placeholder(tf.float32, [None, None])
        self.krP = tf.placeholder(tf.float32)
        self.krX = tf.placeholder(tf.float32)
        self.krH = tf.placeholder(tf.float32)
        self.krR = tf.placeholder(tf.float32)
        
        self.nodes = tf.shape(self.x)[0]
        self.samples = tf.shape(self.x)[1]
        
        # Create embeddings dictionaries
        with tf.variable_scope("RNN", initializer=tf.random_normal_initializer(stddev=0.1)):
            self.L = tf.get_variable("L",
                [self.vocabulary_size, self.word_to_word_embeddings])
        self.L_hat = tf.concat(0, [tf.zeros([1, self.word_to_word_embeddings]), self.L])
        
        # Compute pos features
        P = tf.one_hot(self.p, self.pos_dimension, on_value=10.)
        P = tf.reshape(P, [self.nodes, self.samples, self.poses*self.pos_dimension])
        self.P = tf.nn.dropout(P, self.krP)
        return
    
    def create_word_embedding_layer(self):
        self.word_dimension = self.word_to_word_embeddings
        X = tf.gather(self.L_hat, self.x+1)
        X = tf.reshape(X, [self.nodes, self.samples, self.words*self.word_dimension])
        self.X = tf.nn.dropout(X, self.krX)
        return
    
    def create_hidden_unit(self):
        with tf.variable_scope("RNN", initializer=tf.contrib.layers.xavier_initializer()):
            self.W_h_forward = tf.get_variable("W_h_forward",
                                    [self.pos_dimension * self.poses
                                   + self.word_dimension * self.words
                                   + self.hidden_dimension,
                                     self.hidden_dimension])
            self.W_h_backward = tf.get_variable("W_h_backward",
                                    [self.pos_dimension * self.poses
                                   + self.word_dimension * self.words
                                   + self.hidden_dimension,
                                     self.hidden_dimension])
            self.b_h_forward = tf.get_variable("b_h_forward", [1, self.hidden_dimension])
            self.b_h_backward = tf.get_variable("b_h_backward", [1, self.hidden_dimension])
            
        self.f_h_forward = lambda x: tf.nn.relu(tf.matmul(x, self.W_h_forward)+self.b_h_forward)
        self.f_h_backward = lambda x: tf.nn.relu(tf.matmul(x, self.W_h_backward)+self.b_h_backward)
        return
    
    def create_hidden_layer(self):
        self.create_hidden_unit()
        
        H = tf.zeros([self.nodes+2, self.samples, self.hidden_dimension])
        
        # Forward pass
        def forward_condition(index, H):
            return index <= self.nodes
        def forward_body(index, H):
            p = tf.slice(self.P, [index-1, 0, 0], [1, self.samples, self.poses*self.pos_dimension])
            x = tf.slice(self.X, [index-1, 0, 0], [1, self.samples, self.words*self.word_dimension])
            c = tf.slice(H, [index-1, 0, 0], [1, self.samples, self.hidden_dimension])
            
            # h = self.f_h_forward(tf.concat(1, [x[0,:,:], c[0,:,:]]))
            h = self.f_h_forward(tf.concat(1, [p[0,:,:], x[0,:,:], c[0,:,:]]))
            h = tf.nn.dropout(h, self.krH)
            
            h_left = tf.zeros([index, self.samples, self.hidden_dimension])
            h_right = tf.zeros([self.nodes+1-index, self.samples, self.hidden_dimension])
            return index+1, H+tf.concat(0, [h_left, [h], h_right])
        _, H_forward = tf.while_loop(forward_condition, forward_body, [tf.constant(1), H])
        H_forward = tf.slice(H_forward, [1,0,0], [self.nodes,self.samples,self.hidden_dimension])
        
        # Backward pass
        def backward_condition(index, H):
            return index >= 1
        def backward_body(index, H):
            p = tf.slice(self.P, [index-1, 0, 0], [1, self.samples, self.poses*self.pos_dimension])
            x = tf.slice(self.X, [index-1, 0, 0], [1, self.samples, self.words*self.word_dimension])
            c = tf.slice(H, [index+1, 0, 0], [1, self.samples, self.hidden_dimension])
            
            # h = self.f_h_backward(tf.concat(1, [x[0,:,:], c[0,:,:]]))
            h = self.f_h_backward(tf.concat(1, [p[0,:,:], x[0,:,:], c[0,:,:]]))
            h = tf.nn.dropout(h, self.krR)
            
            h_left = tf.zeros([index, self.samples, self.hidden_dimension])
            h_right = tf.zeros([self.nodes+1-index, self.samples, self.hidden_dimension])
            return index-1, H+tf.concat(0, [h_left, [h], h_right])
        _, H_backward = tf.while_loop(backward_condition, backward_body, [self.nodes, H])
        H_backward = tf.slice(H_backward, [1,0,0], [self.nodes,self.samples,self.hidden_dimension])
        
        self.H = H_forward + H_backward
        # self.H = tf.concat(2, [H_forward, H_backward])
        return
    
    def create_output_unit(self):
        with tf.variable_scope("RNN", initializer=tf.contrib.layers.xavier_initializer()):
            self.W_o = tf.get_variable("W_o",
                            [self.hidden_dimension, self.output_dimension])
            self.b_o = tf.get_variable("b_o", [1, self.output_dimension])
            
        def output_unit(H):
            H = tf.reshape(H, [self.nodes * self.samples, self.hidden_dimension])
            O = tf.matmul(H, self.W_o) + self.b_o
            return O
        
        self.f_o = output_unit
        return
    
    def create_output(self):
        self.create_output_unit()
        
        O = self.f_o(self.H)
        
        Y_hat = tf.nn.softmax(O)
        self.y_hat = tf.reshape(tf.argmax(Y_hat, 1), [self.nodes, self.samples])
        
        e = tf.reshape(self.e, [self.nodes * self.samples])
        y = tf.reshape(self.y, [self.nodes * self.samples])
        Y = tf.one_hot(y, self.output_dimension, on_value=1.)
        self.loss = tf.reduce_sum(e * tf.nn.softmax_cross_entropy_with_logits(O, Y))
        return
    
    def create_update_op(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=self.epsilon)
        self.update_op = optimizer.minimize(self.loss)
        return
    
    def train(self, root_list):
        """ Train on a single tree
        """
        y, p, x, e = conll_utils.get_batch_input(root_list)
        """
        if cnt >= 4: exit()
        if not np.all(y==0):
            global cnt
            cnt += 1
            print "\n\n"
            print root_list[0].text
            print root_list[0].nodes
            print y.shape
            
            print "y"
            print y[:,0]
            print "p"
            print p[:,0,0]
            print "x"
            print x[:,0,0]
            print "e"
            print e[:,0]
        """
        loss, _ = self.sess.run([self.loss, self.update_op],
                    feed_dict={self.y: y, self.p: p, self.x: x, self.e: e,
                               self.krP: self.keep_rate_P, self.krX: self.keep_rate_X,
                               self.krH: self.keep_rate_H, self.krR: self.keep_rate_R})
        return loss
    
    def evaluate(self, root_list, chunk_ne_dict_list, span_set_list, ne_list):
        y, p, x, e = conll_utils.get_batch_input(root_list)
        
        y_hat = self.sess.run(self.y_hat,
                    feed_dict={self.p: p, self.x: x, self.e: e,
                               self.krP: 1.0, self.krX: 1.0,
                               self.krH: 1.0, self.krR: 1.0})
        
        reals = 0.
        positives = 0.
        true_postives = 0.
        
        for sample in range(y.shape[1]): 
            ner_hat = ["NOT_NE"] * y.shape[0]
            for node, label in enumerate(y_hat[:,sample]):
                if y[node][sample] == -1: continue
                ner_hat[node] = ne_list[label]
            
            chunk_y_dict = {}
            
            former_tag = "NOT_NE"
            chunk0 = -1
            for node, tag in enumerate(ner_hat+["NOT_NE"]):
                if tag[-4:] == "body" and tag[:-5] != former_tag:
                    tag = "NOT_NE"
                
                if tag == "NOT_NE" or tag[-4:] == "head":
                    if former_tag != "NOT_NE":
                        chunk = (chunk0, node)
                        # chunk_y_dict[chunk] = former_tag
                        if chunk not in span_set_list[sample]: chunk_y_dict[chunk] = former_tag
                    chunk0 = node
                    
                former_tag = tag if tag=="NOT_NE" else tag[:-5]
                
            """
            chunk0 = -1
            former_tag = "NOT_NE"
            for node, tag in enumerate(ner_hat+["NOT_NE"]):
                if former_tag != tag:
                    if former_tag != "NOT_NE":
                        chunk = (chunk0, node)
                        chunk_y_dict[chunk] = former_tag
                        # if chunk not in span_set_list[sample]: chunk_y_dict[chunk] = former_tag
                    chunk0 = node
                former_tag = tag
            """
            """
            head_ne = None
            head_index = -1
            for text_index, ne in enumerate(ner_hat):
                ne, part = ne[:-7], ne[-7:]
                if part == "NOT_NE":
                    head_ne = None
                    head_index = -1
                elif part == "_single":
                    print "YOLO"
                    chunk_y_dict[(text_index, text_index+1)] = ne
                    head_ne = None
                    head_index = -1
                elif part == "_head":
                    print "YOLO"
                    head_ne = ne
                    head_index = text_index
                elif part == "_body":
                    print "YOLO"
                    if head_ne != ne:
                        head_ne = None
                        head_index = -1
                elif part == "_tail":
                    print "YOLO"
                    if head_ne == ne:
                        chunk_y_dict[(head_index, text_index+1)] = ne
                    head_ne = None
                    head_index = -1
            """
            
            reals += len(chunk_ne_dict_list[sample])
            positives += len(chunk_y_dict)
            for chunk in chunk_y_dict.iterkeys():
                if chunk not in chunk_ne_dict_list[sample]: continue
                if chunk_ne_dict_list[sample][chunk] != chunk_y_dict[chunk]: continue
                true_postives += 1
        return true_postives, positives, reals
    
    def predict(self, root_node):
        y, T, p, x, w, S, chunk_list = conll_utils.get_batch_input([root_node], self.degree)
        
        y_hat = self.sess.run(self.y_hat,
                    feed_dict={self.T: T, 
                               self.p: p, self.x: x, self.w: w,
                               self.S:S,
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

        
