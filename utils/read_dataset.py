from __future__ import print_function
import os
import tensorflow as tf
import glob
import numpy as np
import random
from skimage import color as skco
from skimage import io as skio
import cv2

def get_file_list(data_dir, pattern):
    assert os.path.exists(data_dir), 'Directory {} not found.'.format(data_dir)

    file_list = []
    file_glob = os.path.join(data_dir, pattern)
    file_list.extend(glob.glob(file_glob))

    assert file_list, 'No file found in {}.'.format(file_glob)

    file_list.sort()

    return file_list

def read_dataset(label_value, data_dir, pattern='*', shuffle_seed=None, subset=None, begin=0):

    file_names = get_file_list(data_dir, pattern)
    if shuffle_seed:
        random.seed(shuffle_seed)
        random.shuffle(file_names)
    if subset:
        file_names = file_names[begin:begin+subset]
    instance_num = len(file_names)
    labels = tf.constant(label_value,shape=[instance_num])
    dataset = tf.data.Dataset.from_tensor_slices((file_names,labels))

    print('Read {} instances from {}'.format(instance_num,data_dir))

    return dataset, instance_num

def read_dataset_withmsk(data_dir, pattern, msk_replace, shuffle_seed=None, subset=None):

    image_names = get_file_list(data_dir, pattern)
    if shuffle_seed:
        random.seed(shuffle_seed)
        random.shuffle(image_names)
    if subset:
        image_names = image_names[:subset]
    instance_num = len(image_names)
    label_names = image_names
    for entry in msk_replace:
        label_names = [name.replace(entry[0],entry[1],1) for name in label_names]
    dataset = tf.data.Dataset.from_tensor_slices((image_names,label_names))

    print('Read {} instances from {}'.format(instance_num,data_dir))

    return dataset, instance_num 

def read_image_withmsk(image_name,label_name,outputsize=None,random_flip=False):

    image_string = tf.read_file(image_name)
    image_decoded = tf.div(tf.cast(tf.image.decode_png(image_string,channels=3),tf.float32),255.0)
    label_string = tf.read_file(label_name)
    label_decoded = tf.div(tf.cast(tf.image.decode_png(label_string,channels=1),tf.int32),255)
    if outputsize:
        image_decoded = tf.image.resize_images(image_decoded,outputsize,align_corners=True,method=0)
        label_decoded = tf.cast(tf.image.resize_images(label_decoded,outputsize,align_corners=True,method=0),tf.int32)
    if random_flip:
        uniform_random = tf.random_uniform([3,], 0, 1.0)
        image_decoded = tf.cond(tf.less(uniform_random[0], .5), lambda: tf.image.flip_up_down(image_decoded), lambda: image_decoded)
        image_decoded = tf.cond(tf.less(uniform_random[1], .5), lambda: tf.image.flip_left_right(image_decoded), lambda: image_decoded)
        image_decoded = tf.cond(tf.less(uniform_random[2], .5), lambda: tf.image.transpose_image(image_decoded), lambda: image_decoded)
        label_decoded = tf.cond(tf.less(uniform_random[0], .5), lambda: tf.image.flip_up_down(label_decoded), lambda: label_decoded)
        label_decoded = tf.cond(tf.less(uniform_random[1], .5), lambda: tf.image.flip_left_right(label_decoded), lambda: label_decoded)
        label_decoded = tf.cond(tf.less(uniform_random[2], .5), lambda: tf.image.transpose_image(label_decoded), lambda: label_decoded)
    return image_decoded,label_decoded,image_name

