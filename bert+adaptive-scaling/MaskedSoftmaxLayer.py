import tensorflow as tf
from tensorflow.python.ops import rnn
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import *
from tensorflow.python.util import nest
from Layer import Layer
from MaskLayer import *

class MaskedSoftmaxLayer(Layer):
    
    def __call__(self, inputs,seq_len = None):
        if inputs.dtype.is_integer:
            inputs = tf.cast(inputs,dtype = tf.float32)
        exp_val = tf.exp(inputs)
        if seq_len != None:
            with tf.variable_scope(self.scope) as scope:
                if self.call_cnt ==0:
                    self.mask = MaskLayer(scope = "Mask", reuse = self.reuse)
                self.check_reuse(scope)
                exp_val = self.mask(exp_val,seq_len)
        return exp_val / (self.epsilon + tf.reduce_sum(exp_val,axis = 1,keep_dims = True))
    
    def set_extra_parameters(self,paras =None):
        self.epsilon = 1e-8
        if paras and "epsilon" in paras:
            self.epsilon = paras["epsilon"]
        
if __name__ =="__main__":

    a = tf.Variable([[1,2,3],[4,5,6]])
    mask = MaskedSoftmaxLayer("maskSoftmax")
    output = mask(a,seq_len = [3,2])
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print (sess.run(output))



def f1_reweight_loss(logits,label_ids,positive_idx,negative_idx,correct_class_weight =None,wrong_confusion_matrix = None, label_size = 19):
    softmax_layer = MaskedSoftmaxLayer("softmax_layer")
    probs = softmax_layer(logits)
    
    batch_idx = tf.range(tf.shape(probs)[0])
    label_with_idx = tf.concat([tf.expand_dims(t, 1) for t in [batch_idx,label_ids]], 1)
    golden_prob = tf.gather_nd(probs,label_with_idx)
    m = tf.reduce_sum(positive_idx)  #m为真实为正的个数P，其中positive_idx为
    n = tf.reduce_sum(negative_idx)   #n为真实为负的个数N
    p1 = tf.reduce_sum(positive_idx * golden_prob) #TP
    p2 = tf.reduce_sum(negative_idx * golden_prob)#TN
    beta2 = 1
    neg_weight = p1 / ((beta2 *m)+n-p2 + 1e-8)
    all_one = tf.ones(tf.shape(golden_prob))
    loss_weight = all_one * positive_idx + all_one * neg_weight * negative_idx
    
    loss = - loss_weight * tf.log(golden_prob +1e-8)
    return loss


