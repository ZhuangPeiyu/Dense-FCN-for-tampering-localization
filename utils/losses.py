import numpy as np
import tensorflow as tf

tf_ver = tf.__version__.split('.')
if int(tf_ver[0])<=1 and int(tf_ver[1])<=4:
    softmax_cross_entropy_with_logits = tf.nn.softmax_cross_entropy_with_logits
else:
    softmax_cross_entropy_with_logits = lambda labels=None, logits=None: tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(labels), logits=logits)

def sparse_weighted_softmax_cross_entropy_with_logits(logits, labels, num_classes=2):
    logits = tf.reshape(logits, (-1, num_classes))

    # consturct one-hot label array
    label_flat = tf.reshape(labels, (-1, 1))
    one_hot_labels = tf.reshape(tf.one_hot(label_flat, depth=num_classes), (-1, num_classes))

    # compute weights
    occurrence = tf.cast(tf.bincount(label_flat, minlength=num_classes, maxlength=num_classes), tf.float32)
    class_weight =  tf.cond(tf.equal(tf.count_nonzero(occurrence), num_classes), \
                            lambda: tf.expand_dims(tf.div(tf.reduce_mean(occurrence), occurrence), axis=1), \
                            lambda: tf.ones(shape=[num_classes,1]))
    weights = tf.squeeze(tf.matmul(one_hot_labels, class_weight))

    loss = tf.multiply(weights, softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=logits))
    loss = tf.reduce_mean(loss)
    return loss                                                                         

def focal_loss(logits, labels, gamma=2.0, num_classes=2):
    logits = tf.reshape(logits, (-1, num_classes))

    # consturct one-hot label array
    label_flat = tf.reshape(labels, (-1, 1))
    one_hot_labels = tf.reshape(tf.one_hot(label_flat, depth=num_classes), (-1, num_classes))

    
    # compute weights
    weights = tf.reduce_sum(tf.multiply(one_hot_labels,tf.pow(tf.subtract(1.0, tf.nn.softmax(logits)), gamma)),1)

    loss = tf.multiply(weights, softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=logits))
    loss = tf.reduce_mean(loss)
    return loss

def quasi_f1_loss(logits, labels, num_classes=2):
    logits = tf.nn.softmax(tf.reshape(logits, (-1, num_classes)))

    # consturct one-hot label array
    label_flat = tf.reshape(labels, (-1, 1))
    one_hot_labels = tf.reshape(tf.one_hot(label_flat, depth=num_classes), (-1, num_classes))

    loss = 1-tf.reduce_sum(logits[:,1]*one_hot_labels[:,1])/tf.reduce_sum(logits[:,1]+one_hot_labels[:,1]) \
             -tf.reduce_sum((1-logits[:,0])*(1-one_hot_labels[:,0]))/tf.reduce_sum((1-logits[:,0])+(1-one_hot_labels[:,0]))
    return loss

def sparse_softmax_cross_entropy_with_logits(logits, labels):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits))