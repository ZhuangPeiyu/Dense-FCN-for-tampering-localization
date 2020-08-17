from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import regularizers


def denseFCN(inputs, is_training, weight_decay=5e-4,num_classes=2):

    processed_image = inputs - np.array([123.68, 116.78, 103.94]) / 255.0
    '''Densely block1 '''
    conv1 = tf.layers.conv2d(processed_image,filters = 8, kernel_size = (3,3),kernel_regularizer=regularizers.l2_regularizer(weight_decay),strides=(1,1),padding='SAME',use_bias = True,kernel_initializer = initializers.variance_scaling_initializer(),name = 'conv1')
    norm1 = tf.layers.batch_normalization(conv1,name = 'norm1',training=is_training)
    norm1_activation = tf.nn.relu(norm1, name = 'norm1_activation')

    conv2_input = norm1_activation
    conv2 = tf.layers.conv2d(conv2_input,filters = 8, kernel_size = (3,3),strides=(1,1),kernel_regularizer=regularizers.l2_regularizer(weight_decay),padding='SAME',use_bias=True, kernel_initializer=initializers.variance_scaling_initializer(),name = 'conv2' )
    norm2 = tf.layers.batch_normalization(conv2,name = 'norm2',training=is_training)
    norm2_activation = tf.nn.relu(norm2,name = 'norm2_activation')

    conv3_input = tf.concat([norm1_activation,norm2_activation],axis = -1)
    conv3 = tf.layers.conv2d(conv3_input, filters=8, kernel_size=(3, 3), strides=(1, 1), padding='SAME', use_bias=True,
                             kernel_initializer=initializers.variance_scaling_initializer(), name='conv3',kernel_regularizer=regularizers.l2_regularizer(weight_decay))
    norm3 = tf.layers.batch_normalization(conv3, name='norm3', training=is_training)
    norm3_activation = tf.nn.relu(norm3, name='norm3_activation')

    conv4_input = tf.concat([norm1_activation,norm2_activation,norm3_activation],axis = -1)
    conv4 = tf.layers.conv2d(conv4_input, filters=8, kernel_size=(3, 3), strides=(1, 1), padding='SAME', use_bias=True,
                             kernel_initializer=initializers.variance_scaling_initializer(), name='conv4',kernel_regularizer=regularizers.l2_regularizer(weight_decay))
    norm4 = tf.layers.batch_normalization(conv4, name='norm4', training=is_training)
    norm4_activation = tf.nn.relu(norm4, name='norm4_activation')

    densely_block1_output = tf.concat([norm1_activation,norm2_activation,norm3_activation,norm4_activation],axis = -1)

    block1_transition = tf.layers.conv2d(densely_block1_output,filters = 16, kernel_size = (1,1),padding='SAME',use_bias=True,kernel_initializer=initializers.variance_scaling_initializer(),name = 'block1_transition',kernel_regularizer=regularizers.l2_regularizer(weight_decay))
    
    pool1 = tf.layers.average_pooling2d(block1_transition,pool_size = (2,2),strides = (2,2),padding = 'SAME',name = 'pool1')

    '''Densely block2 '''
    conv5 = tf.layers.conv2d(pool1, filters=16, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                             use_bias=True, kernel_initializer=initializers.variance_scaling_initializer(), name='conv5',kernel_regularizer=regularizers.l2_regularizer(weight_decay))
    norm5 = tf.layers.batch_normalization(conv5, name='norm5', training=is_training)
    norm5_activation = tf.nn.relu(norm5, name='norm5_activation')

    conv6_input = tf.concat([pool1,norm5_activation],axis = -1)
    conv6 = tf.layers.conv2d(conv6_input, filters=16, kernel_size=(3, 3), strides=(1, 1), padding='SAME', use_bias=True,
                             kernel_initializer=initializers.variance_scaling_initializer(), name='conv6',kernel_regularizer=regularizers.l2_regularizer(weight_decay))
    norm6 = tf.layers.batch_normalization(conv6, name='norm6', training=is_training)
    norm6_activation = tf.nn.relu(norm6, name='norm6_activation')

    densely_block2_output = tf.concat([pool1,norm5_activation,norm6_activation],axis=-1)

    block2_transition = tf.layers.conv2d(densely_block2_output, filters=32, kernel_size=(1, 1), padding='SAME',
                                         use_bias=True, kernel_initializer=initializers.variance_scaling_initializer(),
                                         name='block2_transition',kernel_regularizer=regularizers.l2_regularizer(weight_decay))

    pool2 = tf.layers.average_pooling2d(block2_transition, pool_size=(2, 2), strides=(2, 2), padding='SAME',
                                        name='pool2')
    '''Densely block3 '''
    conv7 = tf.layers.conv2d(pool2, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                             use_bias=True, kernel_initializer=initializers.variance_scaling_initializer(), name='conv7',kernel_regularizer=regularizers.l2_regularizer(weight_decay))
    norm7 = tf.layers.batch_normalization(conv7, name='norm7', training=is_training)
    norm7_activation = tf.nn.relu(norm7, name='norm7_activation')

    conv8_input = tf.concat([pool2, norm7_activation], axis=-1)
    conv8 = tf.layers.conv2d(conv8_input, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME', use_bias=True,
                             kernel_initializer=initializers.variance_scaling_initializer(), name='conv8',kernel_regularizer=regularizers.l2_regularizer(weight_decay))
    norm8 = tf.layers.batch_normalization(conv8, name='norm8', training=is_training)
    norm8_activation = tf.nn.relu(norm8, name='norm8_activation')
    densely_block3_output = tf.concat([pool2, norm7_activation, norm8_activation], axis=-1)

    block3_transition = tf.layers.conv2d(densely_block3_output, filters=64, kernel_size=(1, 1), padding='SAME',
                                         use_bias=True, kernel_initializer=initializers.variance_scaling_initializer(),
                                         name='block3_transition',kernel_regularizer=regularizers.l2_regularizer(weight_decay))
    pool3 = tf.layers.average_pooling2d(block3_transition, pool_size=(2, 2), strides=(2, 2), padding='SAME',
                                        name='pool3')

    '''Densely block4 '''
    conv9 = tf.nn.atrous_conv2d(pool3,filters = tf.get_variable("dilate_weight1",shape = [3,3,64,64],initializer=initializers.variance_scaling_initializer()),rate=2,padding='SAME',name = 'conv9')
    norm9 = tf.layers.batch_normalization(conv9, name='norm9', training=is_training)
    norm9_activation = tf.nn.relu(norm9, name='norm9_activation')

    conv10_input = tf.concat([pool3, norm9_activation], axis=-1)
    conv10 = tf.nn.atrous_conv2d(conv10_input,filters =tf.get_variable("dilate_weight2",shape = [3,3,128,64],initializer=initializers.variance_scaling_initializer()),rate = 2,padding = 'SAME',name='conv10')
    norm10 = tf.layers.batch_normalization(conv10, name='conv10', training=is_training)
    norm10_activation = tf.nn.relu(norm10, name='norm10_activation')

    densely_block4_output = tf.concat([pool3, norm9_activation, norm10_activation], axis=-1)

    block4_transition = tf.layers.conv2d(densely_block4_output, filters=96, kernel_size=(1, 1), padding='SAME',
                                         use_bias=True, kernel_initializer=initializers.variance_scaling_initializer(),
                                         name='block4_transition',kernel_regularizer=regularizers.l2_regularizer(weight_decay))
    
    conv11 = tf.nn.atrous_conv2d(block4_transition,filters = tf.get_variable("dilate_weight3",shape = [3,3,96,96],initializer=initializers.variance_scaling_initializer()),rate = 2, padding = 'SAME',name = 'conv11')
    
    norm11 = tf.layers.batch_normalization(conv11, name='norm11', training=is_training)
    norm11_activation = tf.nn.relu(norm11, name='norm11_activation')

    conv12_input = tf.concat([block4_transition,norm11_activation],axis = -1)
    conv12 = tf.nn.atrous_conv2d(conv12_input,filters = tf.get_variable("dilate_weight4",shape = [3,3,192,96],initializer=initializers.variance_scaling_initializer()),rate = 2, padding = 'SAME',name = 'conv12')
    norm12 = tf.layers.batch_normalization(conv12, name='norm12', training=is_training)
    norm12_activation = tf.nn.relu(norm12, name='norm12_activation')

    densely_block5_output = tf.concat([block4_transition,norm11_activation, norm12_activation], axis=-1)
    
    block5_transition = tf.layers.conv2d(densely_block5_output, filters=96, kernel_size=(1, 1), padding='SAME',
                                         use_bias=True, kernel_initializer=initializers.variance_scaling_initializer(),
                                         name='block5_transition',kernel_regularizer=regularizers.l2_regularizer(weight_decay))
    
    '''de_conv #1 '''
    de_conv1_shape = tf.shape(block3_transition)
    net1 = tf.nn.conv2d_transpose(block5_transition, tf.get_variable('bilinear_kernel1', dtype=tf.float32, shape=[4, 4, 64, 96],
                                                         initializer=tf.constant_initializer(bilinear_upsample_weights(2, 64, 96),verify_shape=True),
                                                         regularizer=regularizers.l2_regularizer(weight_decay)),[de_conv1_shape[0], de_conv1_shape[1], de_conv1_shape[2], 64], strides=[1, 2, 2, 1],padding='SAME', name='de_conv1')
    tf.add_to_collection('activations', net1)
    net1 = tf.layers.batch_normalization(net1,training= is_training)
    net1 = tf.nn.relu(net1)

    '''Dense Block 6'''
    conv13 = tf.layers.conv2d(net1, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                             use_bias=True, kernel_initializer=initializers.variance_scaling_initializer(),
                             name='conv13', kernel_regularizer=regularizers.l2_regularizer(weight_decay))
    norm13 = tf.layers.batch_normalization(conv13, name='norm13', training=is_training)
    norm13_activation = tf.nn.relu(norm13, name='norm13_activation')

    conv14_input = tf.concat([net1, norm13_activation], axis=-1)
    conv14 = tf.layers.conv2d(conv14_input, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME', use_bias=True,
                             kernel_initializer=initializers.variance_scaling_initializer(), name='conv14',
                             kernel_regularizer=regularizers.l2_regularizer(weight_decay))
    norm14 = tf.layers.batch_normalization(conv14, name='norm14', training=is_training)
    norm14_activation = tf.nn.relu(norm14, name='norm14_activation')

    densely_block6_output = tf.concat([net1, norm13_activation, norm14_activation], axis=-1)

    block6_transition = tf.layers.conv2d(densely_block6_output, filters=64, kernel_size=(1, 1), padding='SAME',
                                         use_bias=True, kernel_initializer=initializers.variance_scaling_initializer(),
                                         name='block6_transition',
                                         kernel_regularizer=regularizers.l2_regularizer(weight_decay))

    '''de_conv #2'''
    de_conv2_input = block6_transition
    de_conv2_shape = tf.shape(block2_transition)
    net2 = tf.nn.conv2d_transpose(de_conv2_input, tf.get_variable('bilinear_kernel2', dtype=tf.float32, shape=[4, 4, 48, 64],
                                                        initializer=tf.constant_initializer(
                                                            bilinear_upsample_weights(2, 48, 64),
                                                            verify_shape=True),
                                                        regularizer=regularizers.l2_regularizer(weight_decay)),
                                  [de_conv2_shape[0], de_conv2_shape[1], de_conv2_shape[2], 48], strides=[1, 2, 2, 1],
                                  padding='SAME', name='de_conv2')
    tf.add_to_collection('activations', net2)
    net2 = tf.layers.batch_normalization(net2,training= is_training)
    net2 = tf.nn.relu(net2)

    '''Dense Block 7'''
    conv15 = tf.layers.conv2d(net2, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                              use_bias=True, kernel_initializer=initializers.variance_scaling_initializer(),
                              name='conv15', kernel_regularizer=regularizers.l2_regularizer(weight_decay))
    norm15 = tf.layers.batch_normalization(conv15, name='norm15', training=is_training)
    norm15_activation = tf.nn.relu(norm15, name='norm15_activation')

    conv16_input = tf.concat([net2, norm15_activation], axis=-1)
    conv16 = tf.layers.conv2d(conv16_input, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                              use_bias=True,
                              kernel_initializer=initializers.variance_scaling_initializer(), name='conv16',
                              kernel_regularizer=regularizers.l2_regularizer(weight_decay))
    norm16 = tf.layers.batch_normalization(conv16, name='norm16', training=is_training)
    norm16_activation = tf.nn.relu(norm16, name='norm16_activation')

    densely_block7_output = tf.concat([net2, norm15_activation, norm16_activation], axis=-1)

    block7_transition = tf.layers.conv2d(densely_block7_output, filters=48, kernel_size=(1, 1), padding='SAME',
                                         use_bias=True, kernel_initializer=initializers.variance_scaling_initializer(),
                                         name='block7_transition',
                                         kernel_regularizer=regularizers.l2_regularizer(weight_decay))
    '''de_conv #3'''
    de_conv3_input = block7_transition
    de_conv3_shape = tf.shape(block1_transition)
    net3 = tf.nn.conv2d_transpose(de_conv3_input,
                                  tf.get_variable('bilinear_kernel3', dtype=tf.float32, shape=[4, 4, 32, 48],
                                                  initializer=tf.constant_initializer(
                                                      bilinear_upsample_weights(2, 32, 48),
                                                      verify_shape=True),
                                                  regularizer=regularizers.l2_regularizer(weight_decay)),
                                  [de_conv3_shape[0], de_conv3_shape[1], de_conv3_shape[2], 32], strides=[1, 2, 2, 1],
                                  padding='SAME', name='de_conv3')
    tf.add_to_collection('activations', net3)
    net3 = tf.layers.batch_normalization(net3, training=is_training)
    net3 = tf.nn.relu(net3)

    logits_msk = layers.conv2d(net3,num_classes,[5,5],activation_fn=None,normalizer_fn=None, scope='logits')
    preds_msk_map = tf.nn.softmax(logits_msk)[:,:,:,1]
    preds_msk = tf.cast(tf.greater_equal(preds_msk_map,0.5),tf.int32)


    return logits_msk, preds_msk, preds_msk_map
    

def bilinear_upsample_weights(factor, out_channels, in_channels):
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    """
    filter_size = 2 * factor - factor % 2
    center = (factor - 1) if filter_size % 2 == 1 else factor - 0.5
    og = np.ogrid[:filter_size, :filter_size]
    upsample_kernel = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weights = np.tile(upsample_kernel[:,:,np.newaxis,np.newaxis],(1,1,out_channels,in_channels))

    return weights

