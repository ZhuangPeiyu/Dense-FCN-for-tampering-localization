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

def read_image(file_name,label,outputsize=None):
  image_string = tf.read_file(file_name)
  image_decoded = tf.div(tf.cast(tf.image.decode_png(image_string),tf.float32),255.0)
  if outputsize:
        image_decoded = tf.image.resize_images(image_decoded,outputsize,align_corners=False,method=3)
  return image_decoded,label

def read_image_withmsk(image_name,label_name,outputsize=None,random_flip=False):

    image_string = tf.read_file(image_name)
    image_decoded = tf.div(tf.cast(tf.image.decode_png(image_string,channels=3),tf.float32),255.0)
    # image_decoded = tf.image.resize_images(image_decoded,(512,512))

    label_string = tf.read_file(label_name)
    label_decoded = tf.div(tf.cast(tf.image.decode_png(label_string,channels=1),tf.int32),255)
    # # # resize
    image_decoded = tf.image.resize_images(image_decoded, (512,512), align_corners=True, method=0)
    label_decoded = tf.cast(tf.image.resize_images(label_decoded, (512,512), align_corners=True, method=0), tf.int32)
    # crop
    # image_decoded = tf.image.central_crop(image_decoded,0.5)
    # label_decoded = tf.cast(tf.image.central_crop(label_decoded,0.5),tf.int32)
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

def read_image_withmsk_dct(image_name,label_name,outputsize=None,random_flip=False):
    step = 8
    image_string = tf.read_file(image_name)
    image_decoded_dct = tf.cast(tf.image.decode_png(image_string, channels=3),tf.uint8)
    # img_ycbcr = tf.get_default_session().run(image_decoded)
    # img_ycbcr = skco.rgb2ycbcr(img_ycbcr).astype(np.float32)
    #
    # Y = img_ycbcr[:, :, 0]
    # Cb = img_ycbcr[:, :, 1]
    # Cr = img_ycbcr[:, :, 2]
    # row, col = Y.shape
    # mod_row = row % step
    # mod_col = col % step
    # if (mod_row == 0 and mod_col == 0):
    #     Y_new = Y
    #     Cb_new = Cb
    #     Cr_new = Cr
    # if (mod_row != 0 and mod_col == 0):
    #     Y_new = np.zeros((row + (step - mod_row), col), dtype=np.float32)
    #     Cb_new = np.zeros((row + (step - mod_row), col), dtype=np.float32)
    #     Cr_new = np.zeros((row + (step - mod_row), col), dtype=np.float32)
    #     Y_new[:row, :] = Y
    #     Cb_new[:row, :] = Cb
    #     Cr_new[:row, :] = Cr
    # if (mod_row == 0 and mod_col != 0):
    #     Y_new = np.zeros((row, col + (step - mod_col)), dtype=np.float32)
    #     Cb_new = np.zeros((row, col + (step - mod_col)), dtype=np.float32)
    #     Cr_new = np.zeros((row, col + (step - mod_col)), dtype=np.float32)
    #     Y_new[:, :col] = Y
    #     Cb_new[:, :col] = Cb
    #     Cr_new[:, :col] = Cr
    # if (mod_row != 0 and mod_col != 0):
    #     Y_new = np.zeros((row + (step - mod_row), col + (step - mod_col)), dtype=np.float32)
    #     Cb_new = np.zeros((row + (step - mod_row), col + (step - mod_col)), dtype=np.float32)
    #     Cr_new = np.zeros((row + (step - mod_row), col + (step - mod_col)), dtype=np.float32)
    #     Y_new[:row, :col] = Y
    #     Cb_new[:row, :col] = Cb
    #     Cr_new[:row, :col] = Cr
    #
    # Y_dct = cv2.dct(Y_new)
    # Cb_dct = cv2.dct(Cb_new)
    # Cr_dct = cv2.dct(Cr_new)
    # row, col = Y_dct.shape[0], Y_dct.shape[1]
    # channels = 64
    # Y_dct_feature_map = np.zeros((row // step, col // step, channels), dtype=np.float32)
    # Cb_dct_feature_map = np.zeros((row // step, col // step, channels), dtype=np.float32)
    # Cr_dct_feature_map = np.zeros((row // step, col // step, channels), dtype=np.float32)
    #
    # for i in range(0, row // step):
    #     for j in range(0, col // step):
    #         Y_dct_feature_map[i, j, :] = Y_dct[i * step:(i + 1) * step, j * step:(j + 1) * step].reshape(-1, )
    #         Cb_dct_feature_map[i, j, :] = Cb_dct[i * step:(i + 1) * step, j * step:(j + 1) * step].reshape(-1, )
    #         Cr_dct_feature_map[i, j, :] = Cr_dct[i * step:(i + 1) * step, j * step:(j + 1) * step].reshape(-1, )
    # Y_dct_feature_map = tf.cast(Y_dct_feature_map,dtype=tf.float32)
    # Cb_dct_feature_map = tf.cast(Cb_dct_feature_map, dtype=tf.float32)
    # Cr_dct_feature_map = tf.cast(Cr_dct_feature_map, dtype=tf.float32)

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
    return image_decoded,label_decoded,image_name,image_decoded_dct

def read_dataset_with2msk(data_dir, pattern, msk_replace, shuffle_seed=None, subset=None):

    image_names = get_file_list(data_dir, pattern)
    if shuffle_seed:
        random.seed(shuffle_seed)
        random.shuffle(image_names)
    if subset:
        image_names = image_names[:subset]
    instance_num = len(image_names)
    label1_names = image_names
    label2_names = image_names
    for entry in msk_replace[0]:
        label1_names = [name.replace(entry[0],entry[1],1) for name in label1_names]
    for entry in msk_replace[1]:
        label2_names = [name.replace(entry[0],entry[1],1) for name in label2_names]

    dataset = tf.data.Dataset.from_tensor_slices((image_names,label1_names,label2_names))

    print('Read {} instances from {}'.format(instance_num,data_dir))

    return dataset, instance_num 

def read_image_with2msk(image_name,label1_name,label2_name,outputsize=None,random_flip=False):
    image_string = tf.read_file(image_name)
    image_decoded = tf.div(tf.cast(tf.image.decode_png(image_string,channels=3),tf.float32),255.0)
    label1_string = tf.read_file(label1_name)
    label1_decoded = tf.div(tf.cast(tf.image.decode_png(label1_string,channels=1),tf.int32),255)
    label2_string = tf.read_file(label2_name)
    label2_decoded = tf.div(tf.cast(tf.image.decode_png(label2_string,channels=1),tf.int32),255)
    if outputsize:
        image_decoded = tf.image.resize_images(image_decoded,outputsize,align_corners=True,method=0)
        label1_decoded = tf.cast(tf.image.resize_images(label1_decoded,outputsize,align_corners=True,method=0),tf.int32)
        label2_decoded = tf.cast(tf.image.resize_images(label2_decoded,outputsize,align_corners=True,method=0),tf.int32)
    if random_flip:
        uniform_random = tf.random_uniform([3,], 0, 1.0)
        image_decoded = tf.cond(tf.less(uniform_random[0], .5), lambda: tf.image.flip_up_down(image_decoded), lambda: image_decoded)
        image_decoded = tf.cond(tf.less(uniform_random[1], .5), lambda: tf.image.flip_left_right(image_decoded), lambda: image_decoded)
        image_decoded = tf.cond(tf.less(uniform_random[2], .5), lambda: tf.image.transpose_image(image_decoded), lambda: image_decoded)
        label1_decoded = tf.cond(tf.less(uniform_random[0], .5), lambda: tf.image.flip_up_down(label1_decoded), lambda: label1_decoded)
        label1_decoded = tf.cond(tf.less(uniform_random[1], .5), lambda: tf.image.flip_left_right(label1_decoded), lambda: label1_decoded)
        label1_decoded = tf.cond(tf.less(uniform_random[2], .5), lambda: tf.image.transpose_image(label1_decoded), lambda: label1_decoded)
        label2_decoded = tf.cond(tf.less(uniform_random[0], .5), lambda: tf.image.flip_up_down(label2_decoded), lambda: label2_decoded)
        label2_decoded = tf.cond(tf.less(uniform_random[1], .5), lambda: tf.image.flip_left_right(label2_decoded), lambda: label2_decoded)
        label2_decoded = tf.cond(tf.less(uniform_random[2], .5), lambda: tf.image.transpose_image(label2_decoded), lambda: label2_decoded)
    return image_decoded,label1_decoded,label2_decoded,image_name

# if __name__=="__main__":
#     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     os.environ['CUDA_VISIBLE_DEVICES'] = "5"

    # print('\033[7mExample for classification.\033[0m')
    # batch_size = 8
    # dataset0, _ = read_dataset(0,'./data/inpaint_orig_j75/',pattern='*.png',shuffle_seed=1)
    # # dataset0 = dataset0.map(read_image)
    # dataset1, _ = read_dataset(1,'./data/inpaint_fake_j75/',pattern='*.png',shuffle_seed=1)
    # # dataset1 = dataset1.map(read_image)
    # dataset  = tf.data.Dataset.zip((dataset0,dataset1))
    # # dataset = dataset.shuffle(buffer_size=10000)
    # dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size/2))
    # dataset = dataset.repeat()

    # iterator = dataset.make_one_shot_iterator()
    # next_element = iterator.get_next()
    # image_batch = tf.concat([next_element[0][0],next_element[1][0]],0)
    # label_batch = tf.concat([next_element[0][1],next_element[1][1]],0)

    # sess = tf.Session()
    # a,b = sess.run([image_batch,label_batch])
    # print(a)
    # print(b)

    # print('\033[7mExample for classification with pairs.\033[0m')
    # batch_size = 16
    # dataset0, _  = read_dataset(0,'./data/inpaint_orig_j75/',pattern='*.png')
    # dataset0 = dataset0.shuffle(buffer_size=10000)#.map(read_image)
    # dataset1, _  = read_dataset(1,'./data/inpaint_fake_j75/',pattern='*.png')
    # dataset1 = dataset1.shuffle(buffer_size=10000)#.map(read_image)
    # dataset = tf.data.Dataset.zip((dataset0.concatenate(dataset0).concatenate(dataset1).concatenate(dataset1), \
    #                                dataset1.concatenate(dataset0).concatenate(dataset1).concatenate(dataset0))) \
    #                               .map(lambda x,y:(x[0],y[0],(x[1]+y[1])%2)).shuffle(buffer_size=10000) \
    #                               .apply(tf.contrib.data.batch_and_drop_remainder(batch_size)).repeat()

    # iterator = dataset.make_one_shot_iterator()
    # next_element = iterator.get_next()

    # sess = tf.Session()
    # a,b,c = sess.run(next_element)
    # print(a.shape)
    # print(b.shape)
    # print(c)

    # print('\033[7mExample for dataset with masks.\033[0m')
    # dataset, _  = read_dataset_withmsk('./data/full/train/jpg75/TOG/',pattern='*.jpg',msk_replace=[['jpg','msk'],['.jpg','.png']],shuffle_seed=1)

    # iterator = dataset.make_one_shot_iterator()
    # next_element = iterator.get_next()

    # sess = tf.Session()
    # for i in range(10):
    #     a = sess.run(next_element)
    #     print(a)