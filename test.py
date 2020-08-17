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

import denseFCN # Proposed model

FLAGS = tf.flags.FLAGS
#When testing, the batch size is set to be 1#
tf.flags.DEFINE_integer('batch_size', 1, 'batch size')
tf.flags.DEFINE_string('data_dir', './testedImages/NIST-2016/tamper', 'path to dataset')

# tf.flags.DEFINE_string('restore', './Models/FinetuneWithPS-boundary-100/model.ckpt-0.439022-0.80502-6', 'Explicitly restore checkpoint')
# tf.flags.DEFINE_string('restore', './Models/FinetuneWithPS-arbitrary-100/model.ckpt-0.264207-0.741224-3', 'Explicitly restore checkpoint')
tf.flags.DEFINE_string('restore', './Models/FinetuneWithNIST-2016-56/model.ckpt-0.418555-0.753885-8', 'Explicitly restore checkpoint')

tf.flags.DEFINE_string('visout_dir',
                           './Results/NIST-2016/unthresholded','path to output unthresholded predict maps')
tf.flags.DEFINE_string('visout_threshold_dir',
                           './Results/NIST-2016/thresholded',
                       'path to output thresholded predict maps (use 0.5)')
tf.flags.DEFINE_string('record_path',
                           './Results/NIST-2016/metrics',
                       'path to output a recording file ')

if(os.path.exists(FLAGS.visout_dir)==False):
    os.mkdir(FLAGS.visout_dir)
if(os.path.exists(FLAGS.visout_threshold_dir) == False):
    os.mkdir(FLAGS.visout_threshold_dir)
if(os.path.exists(FLAGS.record_path) == False):
    os.mkdir(FLAGS.record_path)
f = open(os.path.join(FLAGS.record_path,"log.txt"),'w+')


'''In testing phase, the following setting is ignored'''
tf.flags.DEFINE_integer('subset', None, 'Use a subset of the whole dataset')
tf.flags.DEFINE_string('img_size', None, 'size of input image')
tf.flags.DEFINE_bool('img_aug', None, 'apply image augmentation')
tf.flags.DEFINE_string('mode', 'test', 'Mode: train / test / visual')
tf.flags.DEFINE_integer('epoch', 30, 'No. of epoch to run')
tf.flags.DEFINE_float('train_ratio', 1.0, 'Trainning ratio')

tf.flags.DEFINE_bool('reset_global_step', True, 'Reset global step')
tf.flags.DEFINE_integer('test_img_num',len(os.listdir(FLAGS.data_dir)),'Test image num')
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

def read_image(image_path,mask_path,image_index):
    imgs = os.listdir(image_path)
    img_name = imgs[image_index]
    images = io.imread(os.path.join(image_path,img_name))
    image_size = images.shape
    row,col,ch = image_size[0],image_size[1],image_size[2]
    if(ch!=3):
        images = Image.open(os.path.join(image_path,img_name)).convert('RGB')
    # The name for
    if ('PS-boundary' in image_path or 'PS-arbitrary' in image_path):
        mask_name = img_name.replace('ps','ms')
        mask_name = mask_name.replace('.jpg','.png')
    elif ('NIST-2016' in image_path):
        mask_name = img_name.replace('PS', 'MS')
    mask_name = mask_name.replace('.jpg', '.png')
    
    print(os.path.join(mask_path,mask_name))
    mask = cv2.imread(os.path.join(mask_path,mask_name),0).astype(dtype = np.uint8)
    mask_copy = np.copy(mask)
    mask[np.where(mask_copy<128)] = 0
    mask[np.where(mask_copy>=128)] = 255
    
    images = np.reshape(images,[1,row,col,3]).astype(dtype=np.float32)/255.0
    mask = np.reshape(mask,[1,row,col]).astype(dtype=np.float32)/255.0
    return images,mask,img_name,mask_name

def main(argv=None):
    t1 = datetime.now()
    print_func = print

    # choose one GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    shuffle_seed = FLAGS.shuffle_seed
    print_func('Seed={}'.format(shuffle_seed))
    if 'PS-boundary' in FLAGS.data_dir or 'PS-arbitrary' in FLAGS.data_dir:
        pattern = "*.jpg"
        msk_rep = [['tamper','masks'],['.jpg','.png'],['ps','ms']]

    if 'NIST-2016' in FLAGS.data_dir:
        pattern = "*.jpg"
        msk_rep = [['tamper','masks'],['PS','MS'],['.jpg','.png']]
    dataset, instance_num = utils.read_dataset.read_dataset_withmsk(FLAGS.data_dir, pattern=pattern,
                                                                    msk_replace=msk_rep, shuffle_seed=shuffle_seed,
                                                                    subset=FLAGS.subset)
    def map_func(*args):
        return utils.read_dataset.read_image_withmsk(*args, outputsize=[int(v) for v in reversed(
            FLAGS.img_size.split('x'))] if FLAGS.img_size else None, random_flip=FLAGS.img_aug)

    if FLAGS.mode == 'test' or FLAGS.mode == 'visual':
        dataset_vld = dataset.map(map_func).batch(FLAGS.batch_size)
        iterator_vld = dataset_vld.make_initializable_iterator()

    is_training = tf.placeholder(tf.bool, [])
    images = tf.placeholder(tf.float32,[None,None,None,3])
    labels_msk = tf.placeholder(tf.int32,[None,None,None])
    imgnames = tf.placeholder(tf.string,[])
    logits_msk, preds_msk, preds_msk_map = model(images, FLAGS.weight_decay, is_training)  # pylint: disable=W0612

    itr_per_epoch = int(np.ceil(instance_num * FLAGS.train_ratio) / FLAGS.batch_size)
    print("itr_per_epoch " + str(itr_per_epoch))

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
        
        index = 0
        f1_avg = 0.0
        recall_avg = 0.0
        prec_avg = 0.0
        tnr_avg = 0.0
        iou_avg = 0.0
        mcc_avg = 0.0
        auc_avg = 0.0
        try:
            image_path = FLAGS.data_dir
            mask_path = image_path
            for entry in msk_rep:
                mask_path = mask_path.replace(entry[0],entry[1])
            count = 0
            for image_index in range(FLAGS.test_img_num):
                try:
                    images_, labels_msk_, imgnames_,mask_name = read_image(image_path, mask_path, image_index)

                    logits_msk_,preds_msk_,preds_msk_map_= sess.run([logits_msk, preds_msk, preds_msk_map],feed_dict={is_training: False,images:images_,labels_msk:labels_msk_,imgnames:imgnames_})

                    image_shape = images_.shape
                    row = image_shape[1]
                    col = image_shape[2]

                    final_predit_mask_map = np.zeros((row,col))
                    final_predit_mask_map+=preds_msk_map_[0]
                    
                    num_every_pixel_scanned = np.ones((row,col))

                    for i in range(FLAGS.batch_size):

                        image = np.copy(images_[0])
                        mask = np.copy(labels_msk_[0])
                        rotate_angle = [180]
                        recovery_angle = [-180]
                        filp_axis = [0,1]
                        save_imgname = str(imgnames_)
                        save_imgname = save_imgname.replace('ps', 'ms')
                        '''Rotate 180'''
                        for angle in range(len(rotate_angle)):
                            # print(image.shape)
                            test_image = transform.rotate(image,angle = rotate_angle[angle])
                            # print(test_image.shape)
                            test_masks = transform.rotate(mask,angle = rotate_angle[angle])
                            test_image = test_image[np.newaxis,:]
                            test_masks = test_masks[np.newaxis,:]
                            preds_, preds_map_ = sess.run([preds_msk, preds_msk_map],
                                                          feed_dict={is_training: False, images: test_image,
                                                                     labels_msk: test_masks, imgnames: imgnames_})

                            final_predit_mask_map+= transform.rotate(preds_map_[0],angle = recovery_angle[angle])

                            num_every_pixel_scanned+=1
                        '''filp'''
                        for axis in filp_axis:
                            test_image = np.flip(image,axis = axis)
                            test_masks = np.flip(mask,axis = axis)
                            test_image = test_image[np.newaxis, :]
                            test_masks = test_masks[np.newaxis, :]
                            preds_, preds_map_ = sess.run([preds_msk, preds_msk_map],
                                                          feed_dict={is_training: False, images: test_image,
                                                                     labels_msk: test_masks, imgnames: imgnames_})

                            final_predit_mask_map += np.flip(preds_map_[0], axis = axis)
                            num_every_pixel_scanned += 1
                        test_image = np.transpose(image,axes=[1,0,2])
                        test_masks = np.transpose(mask)
                        test_image = test_image[np.newaxis, :]
                        test_masks = test_masks[np.newaxis, :]
                        preds_, preds_map_ = sess.run([preds_msk, preds_msk_map],
                                                      feed_dict={is_training: False, images: test_image,
                                                                 labels_msk: test_masks, imgnames: imgnames_})

                        final_predit_mask_map += np.transpose(preds_map_[0],axes = [1,0])
                        num_every_pixel_scanned += 1

                        final_predit_mask_map = final_predit_mask_map/num_every_pixel_scanned
                    '''Compute metrics'''
                    groundtruth = cv2.imread(os.path.join(mask_path, mask_name), 0).astype(np.uint8)
                    groundtruth_copy = np.copy(groundtruth)
                    # the values in some masks are 0 or 1, sometimes the values in some masks are 0 or 255
                    if (np.max(groundtruth)>1):
                        groundtruth[np.where(groundtruth_copy < 128)] = 0
                        groundtruth[np.where(groundtruth_copy >= 128)] = 1
                    else:
                        groundtruth[np.where(groundtruth_copy < 0.5)] = 0
                        groundtruth[np.where(groundtruth_copy >= 0.5)] = 1

                    # compute auc score #
                    auc_result = roc_auc_score(groundtruth.reshape(-1,),final_predit_mask_map.reshape(-1,))

                    save_imgname = str(imgnames_)
                    save_imgname = save_imgname.replace('ps','ms')
					# save the unthreshold predict maps #
                    io.imsave(os.path.join(FLAGS.visout_dir, save_imgname.replace('.jpg', '.png')),np.uint8(np.round(final_predit_mask_map * 255.0)))

                    preds_msk_ = np.copy(final_predit_mask_map)
                    preds_msk_[np.where(final_predit_mask_map<0.5)] = 0
                    preds_msk_[np.where(final_predit_mask_map>=0.5)] = 1
                    # save the thresholded predict maps #
                    io.imsave(os.path.join(FLAGS.visout_threshold_dir, save_imgname.replace('.jpg', '.png')),np.uint8(np.round(preds_msk_ * 255.0)))
                    recall, tnr, prec, f1, mcc, iou, tn, tp, fn, fp = utils.metrics.get_metrics(groundtruth, preds_msk_)
                    if(tnr==np.nan):
                        tnr = 0
                    print('{}: {} '.format(index, save_imgname), end='',file = f)
                    print(
                        'Recall: {:g} Prec: {:g} TNR: {:g} \033[1;31mF1: {:g}\033[0m MCC: {:g} IoU: {:g} AUC:{:g}'.format(recall,
                                                                                                                 prec,
                                                                                                                 tnr,
                                                                                                                 f1,
                                                                                                                 mcc,
                                                                                                                 iou,
                                                                                                                auc_result),
                        end='',file = f)
                    print('',file = f)
                    print('{}: {} '.format(index, save_imgname), end='')
                    
                    print(
                        'Recall: {:g} Prec: {:g} TNR: {:g} \033[1;31mF1: {:g}\033[0m MCC: {:g} IoU: {:g} AUC: {:g}'.format(recall,
                                                                                                                 prec,
                                                                                                                 tnr,
                                                                                                                 f1,
                                                                                                                 mcc,
                                                                                                                 iou,
                                                                                                                auc_result),
                        end='')
                    print('')

                    index+=1
                    f1_avg += f1
                    recall_avg+=recall
                    prec_avg+=prec
                    tnr_avg += tnr
                    iou_avg+=iou
                    mcc_avg+=mcc
                    auc_avg += auc_result
                    count +=1
                    print("Total avg F1: ", str(f1_avg/count), "Total avg MCC : ",str(mcc_avg/count),
                          "Total avg IOU: ", str(iou_avg/count), "Total avg AUC : ",str(auc_avg/count),
                          "Total avg tnr: ",str(tnr_avg/count), "Total avg recall: ",str(recall_avg/count),
                          "Total avg prec: ",str(prec_avg/count))

                except Exception as e:
                    print(e)
                    # continue
                    # print(str(count))

            t2 = datetime.now()
            total_time = (t2-t1).seconds
            print(FLAGS.data_dir)
            print("Total time: ", total_time)
            print('avg f1', str(f1_avg / count))
            print('recall_avg ', str(recall_avg / count))
            print('prec_avg ', str(prec_avg / count))
            print('tnr_avg ', str(tnr_avg / count))
            print('mcc_avg ', str(mcc_avg / count))
            print('iou_avg ', str(iou_avg / count))
            print('auc_avg ', str(auc_avg / count))
            print('avg f1', str(f1_avg / count), file=f)
            print('recall_avg ', str(recall_avg / count), file=f)
            print('prec_avg ', str(prec_avg / count), file=f)
            print('tnr_avg ', str(tnr_avg / count), file=f)
            print('mcc_avg ', str(mcc_avg / count), file=f)
            print('iou_avg ', str(iou_avg / count), file=f)
            print('auc_avg ', str(auc_avg / count), file=f)
        except tf.errors.OutOfRangeError:
            # break
            print("error")
    else:
        print_func('Mode not defined: ' + FLAGS.mode)
        return None


if __name__ == '__main__':
    tf.app.run()
