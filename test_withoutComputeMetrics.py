from __future__ import print_function
import os
import warnings
import numpy as np
import tensorflow as tf
from datetime import datetime

slim = tf.contrib.slim
from skimage import io
import utils
from PIL import Image
import cv2
from skimage import transform
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import denseFCN  # Proposed model

FLAGS = tf.flags.FLAGS
# When testing, the batch size is set to be 1#
tf.flags.DEFINE_integer('batch_size', 1, 'batch size')

tf.flags.DEFINE_string('data_dir',
                       '.\\testedImages\\NIST-2016\\tamper\\',
                       'path to dataset')
tf.flags.DEFINE_string('restore', '.\\Models\\FinetuneWithNIST-2016-56\\model.ckpt-0.418555-0.753885-8', 'Explicitly restore checkpoint')

threshold = 0.5
tf.flags.DEFINE_string('visout_dir',
                       '..\\results\\unthresholded\\',
                       'path to output unthresholded predict maps')
tf.flags.DEFINE_string('visout_threshold_dir',
                       '.\\results\\thresholded_0.5\\',
                       'path to output thresholded predict maps (use 0.5)')
tf.flags.DEFINE_string('record_path',
                       '.\\results\\metrics\\',
                       'path to output a recording file ')


if (os.path.exists(FLAGS.visout_dir) == False):
	os.makedirs(FLAGS.visout_dir)
if (os.path.exists(FLAGS.visout_threshold_dir) == False):
	os.makedirs(FLAGS.visout_threshold_dir)
if (os.path.exists(FLAGS.record_path) == False):
	os.makedirs(FLAGS.record_path)
f = open(os.path.join(FLAGS.record_path, "log.txt"), 'w+')

'''In testing phase, the following setting is ignored'''
tf.flags.DEFINE_integer('subset', None, 'Use a subset of the whole dataset')
tf.flags.DEFINE_string('img_size', None, 'size of input image')
tf.flags.DEFINE_bool('img_aug', None, 'apply image augmentation')
tf.flags.DEFINE_string('mode', 'test', 'Mode: train / test / visual')
tf.flags.DEFINE_integer('epoch', 30, 'No. of epoch to run')
tf.flags.DEFINE_float('train_ratio', 1.0, 'Trainning ratio')

tf.flags.DEFINE_bool('reset_global_step', True, 'Reset global step')
tf.flags.DEFINE_integer('test_img_num', len(os.listdir(FLAGS.data_dir)), 'Test image num')
# learning configuration
tf.flags.DEFINE_string('optimizer', 'Adam', 'GradientDescent / Adadelta / Momentum / Adam / Ftrl / RMSProp')
tf.flags.DEFINE_float('learning_rate', 5e-4, 'Learning rate for Optimizer')
tf.flags.DEFINE_float('lr_decay', 0.5, 'Decay of learning rate')
tf.flags.DEFINE_float('lr_decay_freq', 1.0, 'Epochs that the lr is reduced once')
tf.flags.DEFINE_string('loss', 'xent', 'Loss function type')
tf.flags.DEFINE_float('focal_gamma', '2.0', 'gamma of focal loss')
tf.flags.DEFINE_float('weight_decay', 5e-4, 'Learning rate for Optimizer')
tf.flags.DEFINE_integer('shuffle_seed', None, 'Seed for shuffling images')
tf.flags.DEFINE_integer('verbose_time', 20, 'verbose times in each epoch')
tf.flags.DEFINE_integer('valid_time', 1, 'validation times in each epoch')
tf.flags.DEFINE_integer('keep_ckpt', 0, 'num of checkpoint files to keep')

print("Batch size:", str(FLAGS.batch_size), " , optimizer: ", FLAGS.optimizer, ", Learning rate: ",
      str(FLAGS.learning_rate),
      ", lr decay: ", str(FLAGS.lr_decay), " , Lr decay freq: ", str(FLAGS.lr_decay_freq), " , loss: " + FLAGS.loss)

OPTIMIZERS = {
	'GradientDescent': {'func': tf.train.GradientDescentOptimizer, 'args': {}},
	'Adadelta': {'func': tf.train.AdadeltaOptimizer, 'args': {}},
	'Momentum': {'func': tf.train.MomentumOptimizer, 'args': {'momentum': 0.9}},
	'Adam': {'func': tf.train.AdamOptimizer, 'args': {}},
	'Ftrl': {'func': tf.train.FtrlOptimizer, 'args': {}},
	'RMSProp': {'func': tf.train.RMSPropOptimizer, 'args': {}}
}
LOSS = {
	'wxent': {'func': utils.losses.sparse_weighted_softmax_cross_entropy_with_logits, 'args': {}},
	'focal': {'func': utils.losses.focal_loss, 'args': {'gamma': FLAGS.focal_gamma}},
	'f1': {'func': utils.losses.quasi_f1_loss, 'args': {}},
	'xent': {'func': utils.losses.sparse_softmax_cross_entropy_with_logits, 'args': {}}
}


def model(images, weight_decay, is_training, num_classes=2):
	return denseFCN.denseFCN(images, is_training)
	# return denseFCN.denseFCN_highpass30(images,is_training)

def read_image(image_path, mask_path, image_index):
	imgs = os.listdir(image_path)
	img_name = imgs[image_index]
	mask_name = img_name
	images = io.imread(os.path.join(image_path, img_name))
	image_size = images.shape
	row, col, ch = image_size[0], image_size[1], image_size[2]
	if (ch != 3):
		images = Image.open(os.path.join(image_path, img_name)).convert('RGB')
	# The name for
	if ('PS-boundary' in image_path or 'PS-arbitrary' in image_path):
		mask_name = img_name.replace('ps', 'ms')
		mask_name = mask_name.replace('.jpg', '.png')
	elif ('NIST-2016' in image_path):
		mask_name = img_name.replace('PS', 'MS')
	mask_name = mask_name.replace('.jpg', '.png')

	print(os.path.join(mask_path, mask_name))
	mask = cv2.imread(os.path.join(mask_path, mask_name), 0).astype(dtype=np.uint8)
	mask_copy = np.copy(mask)
	mask[np.where(mask_copy < 128)] = 0
	mask[np.where(mask_copy >= 128)] = 255

	images = np.reshape(images, [1, row, col, 3]).astype(dtype=np.float32) / 255.0
	mask = np.reshape(mask, [1, row, col]).astype(dtype=np.float32) / 255.0
	return images, mask, img_name, mask_name

def read_image_without_mask(image_path,image_index):
	imgs = os.listdir(image_path)
	img_name = imgs[image_index]
	images = io.imread(os.path.join(image_path, img_name))
	image_size = images.shape
	row, col, ch = image_size[0], image_size[1], image_size[2]
	if (ch != 3):
		images = Image.open(os.path.join(image_path, img_name)).convert('RGB')
	# The name for


	images = np.reshape(images, [1, row, col, 3]).astype(dtype=np.float32) / 255.0
	return images,img_name


def main(argv=None):
	print_func = print

	# choose one GPU
	os.environ['CUDA_VISIBLE_DEVICES'] = '1'

	shuffle_seed = FLAGS.shuffle_seed
	print_func('Seed={}'.format(shuffle_seed))

	is_training = tf.placeholder(tf.bool, [])
	images = tf.placeholder(tf.float32, [None, None, None, 3])
	imgnames = tf.placeholder(tf.string, [])
	logits_msk, preds_msk, preds_msk_map = model(images, FLAGS.weight_decay, is_training)  # pylint: disable=W0612

	# itr_per_epoch = int(np.ceil(instance_num * FLAGS.train_ratio) / FLAGS.batch_size)
	# print("itr_per_epoch " + str(itr_per_epoch))

	config = tf.ConfigProto(log_device_placement=False)
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())

	saver = tf.train.Saver(max_to_keep=FLAGS.keep_ckpt + 1 if FLAGS.keep_ckpt else 1000000)
	model_checkpoint_path = ''
	if FLAGS.restore and 'ckpt' in FLAGS.restore:
		model_checkpoint_path = FLAGS.restore
	else:
		ckpt = tf.train.get_checkpoint_state(FLAGS.restore or FLAGS.logdir)
		if ckpt and ckpt.model_checkpoint_path:
			model_checkpoint_path = ckpt.model_checkpoint_path
			model_checkpoint_path = model_checkpoint_path.replace('//', '/')

	if model_checkpoint_path:
		try:
			saver.restore(sess, model_checkpoint_path)
		except tf.errors.NotFoundError:  # compatible code
			variables_to_restore = {var.op.name.replace("global_step", "Variable"): var for var in
			                        tf.global_variables()}
			restorer = tf.train.Saver(variables_to_restore)
			restorer.restore(sess, model_checkpoint_path)
		print_func('Model restored from {}'.format(model_checkpoint_path))

	if FLAGS.mode == 'test':
		warnings.simplefilter('ignore', (UserWarning, RuntimeWarning))

		try:
			image_path = FLAGS.data_dir
			for image_index in tqdm(range(FLAGS.test_img_num)):
				try:
					print(image_index)
					# images_, labels_msk_, imgnames_, mask_name = read_image(image_path, mask_path, image_index)
					images_, imgnames_ = read_image_without_mask(image_path, image_index)
					logits_msk_, preds_msk_, preds_msk_map_ = sess.run([logits_msk, preds_msk, preds_msk_map],
					                                                   feed_dict={is_training: False, images: images_,
					                                                              imgnames: imgnames_})

					image_shape = images_.shape
					row = image_shape[1]
					col = image_shape[2]

					final_predit_mask_map = np.zeros((row, col))
					final_predit_mask_map += preds_msk_map_[0]

					num_every_pixel_scanned = np.ones((row, col))

					for i in range(FLAGS.batch_size):

						image = np.copy(images_[0])

						rotate_angle = [180]
						recovery_angle = [-180]
						filp_axis = [0, 1]
						save_imgname = str(imgnames_)
						save_imgname = save_imgname.replace('ps', 'ms')
						'''Rotate 180'''
						for angle in range(len(rotate_angle)):
							# print(image.shape)
							test_image = transform.rotate(image, angle=rotate_angle[angle])
							# print(test_image.shape)
							test_image = test_image[np.newaxis, :]
							preds_, preds_map_ = sess.run([preds_msk, preds_msk_map],
							                              feed_dict={is_training: False, images: test_image,imgnames: imgnames_})

							final_predit_mask_map += transform.rotate(preds_map_[0], angle=recovery_angle[angle])

							num_every_pixel_scanned += 1
						'''filp'''
						for axis in filp_axis:
							test_image = np.flip(image, axis=axis)
							# test_masks = np.flip(mask, axis=axis)
							test_image = test_image[np.newaxis, :]
							preds_, preds_map_ = sess.run([preds_msk, preds_msk_map],
							                              feed_dict={is_training: False, images: test_image,imgnames: imgnames_})

							final_predit_mask_map += np.flip(preds_map_[0], axis=axis)
							num_every_pixel_scanned += 1
						'''Transposed'''
						test_image = np.transpose(image, axes=[1, 0, 2])
						test_image = test_image[np.newaxis, :]
						preds_, preds_map_ = sess.run([preds_msk, preds_msk_map],
						                              feed_dict={is_training: False, images: test_image,imgnames: imgnames_})

						final_predit_mask_map += np.transpose(preds_map_[0], axes=[1, 0])
						num_every_pixel_scanned += 1

						final_predit_mask_map = final_predit_mask_map / num_every_pixel_scanned

					# save the unthreshold predict maps #
					io.imsave(os.path.join(FLAGS.visout_dir, save_imgname.replace('.jpg', '.png')),
					          np.uint8(np.round(final_predit_mask_map * 255.0)))

					preds_msk_ = np.copy(final_predit_mask_map)
					preds_msk_[np.where(final_predit_mask_map <= threshold)] = 0
					preds_msk_[np.where(final_predit_mask_map > threshold)] = 1
					# save the thresholded predict maps #
					io.imsave(os.path.join(FLAGS.visout_threshold_dir, save_imgname.replace('.jpg', '.png')),
					          np.uint8(np.round(preds_msk_ * 255.0)))
					print(image_index,save_imgname,file = f)
				except Exception as e:
					print(e)
				# continue
				# print(str(count))

		except tf.errors.OutOfRangeError:
			# break
			print("error")
	else:
		print_func('Mode not defined: ' + FLAGS.mode)
		return None
	f.close()

if __name__ == '__main__':
	tf.app.run()
