from __future__ import print_function
import os
import sys
import time
import warnings
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
slim = tf.contrib.slim

from shutil import rmtree
from six.moves import xrange
import utils
import denseFCN

FLAGS = tf.flags.FLAGS

# path to training dataset
tf.flags.DEFINE_string('data_dir',
                   './dataset/train/tamper/',
                   'path to dataset')
# path to validation dataset
vld_dataset = './dataset/val/tamper/'


tf.flags.DEFINE_integer('subset', None, 'Use a subset of the whole dataset')
tf.flags.DEFINE_string('img_size', None, 'size of input image')
tf.flags.DEFINE_bool('img_aug', None, 'apply image augmentation')
tf.flags.DEFINE_string('mode', 'train', 'Mode: train / test / visual')
tf.flags.DEFINE_integer('epoch', 10, 'No. of epoch to run')
tf.flags.DEFINE_float('train_ratio', 1.0, 'Trainning ratio')
tf.flags.DEFINE_string('restore', '', 'Explicitly restore checkpoint')

tf.flags.DEFINE_bool('reset_global_step', True, 'Reset global step')
tf.flags.DEFINE_integer('batch_size', 16, 'batch size')
tf.flags.DEFINE_string('optimizer', 'Adam', 'GradientDescent / Adadelta / Momentum / Adam / Ftrl / RMSProp')
tf.flags.DEFINE_float('learning_rate', 5e-4, 'Learning rate for Optimizer')

tf.flags.DEFINE_float('lr_decay', 0.5, 'Decay of learning rate')
tf.flags.DEFINE_float('lr_decay_freq', 1.0, 'Epochs that the lr is reduced once')
tf.flags.DEFINE_string('loss', 'xent', 'Loss function type')
tf.flags.DEFINE_float('focal_gamma', '2.0', 'gamma of focal loss')
tf.flags.DEFINE_float('weight_decay', 5e-4, 'Learning rate for Optimizer')
tf.flags.DEFINE_integer('shuffle_seed',None, 'Seed for shuffling images')
tf.flags.DEFINE_string('logdir','./Models/','path to save model and log')
tf.flags.DEFINE_integer('verbose_time', 20, 'verbose times in each epoch')
tf.flags.DEFINE_integer('valid_time', 1, 'validation times in each epoch')
tf.flags.DEFINE_integer('keep_ckpt', 0, 'num of checkpoint files to keep, 0 means to save all models')


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


def main(argv=None):
    
    if FLAGS.mode == 'train':
        write_log_mode = 'w'
        if os.path.isdir(FLAGS.logdir) and os.listdir(FLAGS.logdir):
            sys.stderr.write('Log dir is not empty, continue? [y/r/N]: ')
            chioce = input('')
            if (chioce == 'y' or chioce == 'Y'):
                write_log_mode = 'a'
            elif (chioce == 'r' or chioce == 'R'):
                rmtree(FLAGS.logdir)
            else:
                sys.stderr.write('Abort.\n')
                return None
    
    tee_print = utils.tee_print.TeePrint(filename=FLAGS.logdir + 'log.log', mode=write_log_mode)
    print_func = tee_print.write
    print_func("Batch size:{:}, optimizer: {:}, Learning rate:{:}, lr_decay:{:}, lr_decay_freq: {:},loss:{:}, dataset:{:}".format(
        str(FLAGS.batch_size), str(FLAGS.optimizer), str(FLAGS.learning_rate), str(FLAGS.lr_decay),
        str(FLAGS.lr_decay_freq), str(FLAGS.loss), str(FLAGS.data_dir)))
    print_func("saving path: ", FLAGS.logdir)
    
	# choose one GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Setting up dataset
    shuffle_seed = FLAGS.shuffle_seed or np.long(time.time() * 256)
    print_func('Seed={}'.format(shuffle_seed))
    
    
    if 'tamper' in FLAGS.data_dir:
        # the format of the training tampering images
        pattern_train = '*.png'
        # the replacement method of the tampering images and their grouth-truths
        msk_rep_train = [['tamper', 'masks'], ['ps', 'ms']]
        # the format of the validation tampering images #
        pattern_val = "*.jpg"
        # the replacement method of the tampering images and their grouth-truths #
        msk_rep_val = [['.jpg','.png'],['tamper', 'masks'], ['ps', 'ms']]
    
    dataset, instance_num = utils.read_dataset.read_dataset_withmsk(FLAGS.data_dir, pattern=pattern_train,
                                                                    msk_replace=msk_rep_train, shuffle_seed=shuffle_seed,
                                                                    subset=FLAGS.subset)
    dataset2, instance_num2 = utils.read_dataset.read_dataset_withmsk(vld_dataset, pattern=pattern_val,
                                                                      msk_replace=msk_rep_val,
                                                                      shuffle_seed=shuffle_seed,
                                                                      subset=FLAGS.subset)

    def map_func(*args):
        return utils.read_dataset.read_image_withmsk(*args, outputsize=[int(v) for v in reversed(
            FLAGS.img_size.split('x'))] if FLAGS.img_size else None, random_flip=FLAGS.img_aug)

    if FLAGS.mode == 'train':
        dataset_trn = dataset.take(int(np.ceil(instance_num * FLAGS.train_ratio))).shuffle(buffer_size=10000).map(
            map_func).batch(FLAGS.batch_size).repeat()
        dataset_vld = dataset2.map(map_func).batch(1)
        iterator_trn = dataset_trn.make_one_shot_iterator()
        iterator_vld = dataset_vld.make_initializable_iterator()
    elif FLAGS.mode == 'test' or FLAGS.mode == 'visual':
        dataset_vld = dataset.map(map_func).batch(FLAGS.batch_size)
        iterator_vld = dataset_vld.make_initializable_iterator()

    handle = tf.placeholder(tf.string, shape=[])
    if FLAGS.mode == 'train':
        iterator = tf.data.Iterator.from_string_handle(handle, dataset_trn.output_types, dataset_trn.output_shapes)
    if FLAGS.mode == 'test':
        iterator = tf.data.Iterator.from_string_handle(handle, dataset_vld.output_types, dataset_vld.output_shapes)
    next_element = iterator.get_next()
    images = next_element[0]
    
    labels_msk = tf.squeeze(next_element[1], axis=3)

    is_training = tf.placeholder(tf.bool, [])
    logits_msk, preds_msk, preds_msk_map = model(images, FLAGS.weight_decay, is_training)

    loss = LOSS[FLAGS.loss]['func'](logits=logits_msk, labels=labels_msk, **LOSS[FLAGS.loss]['args'])

    global_step = tf.Variable(0, trainable=False, name='global_step')
    itr_per_epoch = int(np.ceil(instance_num * FLAGS.train_ratio) / FLAGS.batch_size)
    print("itr_per_epoch " + str(itr_per_epoch))
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                                               decay_steps=int(itr_per_epoch * FLAGS.lr_decay_freq),
                                               decay_rate=FLAGS.lr_decay, staircase=True)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = OPTIMIZERS[FLAGS.optimizer]['func'](learning_rate, **OPTIMIZERS[FLAGS.optimizer]['args']). \
            minimize(loss, global_step=global_step, var_list=tf.trainable_variables())

    with tf.name_scope('metrics'):
        tp_count = tf.reduce_sum(tf.to_float(tf.logical_and(tf.equal(labels_msk, 1), tf.equal(preds_msk, 1))),
                                 name='true_positives')
        tn_count = tf.reduce_sum(tf.to_float(tf.logical_and(tf.equal(labels_msk, 0), tf.equal(preds_msk, 0))),
                                 name='true_negatives')
        fp_count = tf.reduce_sum(tf.to_float(tf.logical_and(tf.equal(labels_msk, 0), tf.equal(preds_msk, 1))),
                                 name='false_positives')
        fn_count = tf.reduce_sum(tf.to_float(tf.logical_and(tf.equal(labels_msk, 1), tf.equal(preds_msk, 0))),
                                 name='false_negatives')
        metrics_count = tf.Variable(0.0, name='metrics_count', trainable=False,
                                    collections=[tf.GraphKeys.LOCAL_VARIABLES])
        recall_sum = tf.Variable(0.0, name='recall_sum', trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
        precision_sum = tf.Variable(0.0, name='precision_sum', trainable=False,
                                    collections=[tf.GraphKeys.LOCAL_VARIABLES])
        accuracy_sum = tf.Variable(0.0, name='accuracy_sum', trainable=False,
                                   collections=[tf.GraphKeys.LOCAL_VARIABLES])
        loss_sum = tf.Variable(0.0, name='loss_sum', trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
        update_recall_sum = tf.assign_add(recall_sum, tf.cond(tf.equal(tp_count + fn_count, 0), \
                                                              lambda: 0.0, \
                                                              lambda: tp_count / (tp_count + fn_count)))
        update_precision_sum = tf.assign_add(precision_sum, tf.cond(tf.equal(tp_count + fp_count, 0), \
                                                                    lambda: 0.0, \
                                                                    lambda: tp_count / (tp_count + fp_count)))
        update_accuracy_sum = tf.assign_add(accuracy_sum,
                                            (tp_count + tn_count) / (tp_count + tn_count + fp_count + fn_count))
        update_loss_sum = tf.assign_add(loss_sum, loss)
        with tf.control_dependencies([update_recall_sum, update_precision_sum, update_accuracy_sum, update_loss_sum]):
            update_metrics_count = tf.assign_add(metrics_count, 1.0)
        mean_recall = recall_sum / metrics_count
        mean_precision = precision_sum / metrics_count
        mean_accuracy = accuracy_sum / metrics_count
        mean_loss = loss_sum / metrics_count

    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    local_vars_metrics = [v for v in tf.local_variables() if 'metrics/' in v.name]

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

    if FLAGS.mode == 'train':
        summary_op = tf.summary.merge([tf.summary.scalar('loss', mean_loss),
                                       tf.summary.scalar('lr', learning_rate)])
        summary_writer_trn = tf.summary.FileWriter(FLAGS.logdir + '/train', sess.graph)
        summary_writer_vld = tf.summary.FileWriter(FLAGS.logdir + '/validation')

        handle_trn = sess.run(iterator_trn.string_handle())
        handle_vld = sess.run(iterator_vld.string_handle())
        best_metric = 0.0
        best_auc_metric = 0.0
        best_val_loss = 1000000
        loss_decrease = 0.0
        if FLAGS.reset_global_step:
            sess.run(tf.variables_initializer([global_step]))
        for itr in xrange(itr_per_epoch * FLAGS.epoch):  # pylint: disable=W0612
            _, step, _, = sess.run([train_op, global_step, update_metrics_count],
                                   feed_dict={handle: handle_trn, is_training: True})
            if step % (int(itr_per_epoch / FLAGS.verbose_time)) == 0:
                mean_loss_, mean_accuracy_, mean_recall_, mean_precision_, summary_str = sess.run(
                    [mean_loss, mean_accuracy, mean_recall, mean_precision, summary_op])

                print_func('epoch: {} step: {:d} loss: {:g} ACC: {:g} Recall: {:g} Precision: {:g}'.format( \
                    str(int(step / itr_per_epoch)), step, mean_loss_, mean_accuracy_, mean_recall_, mean_precision_))
                summary_writer_trn.add_summary(summary_str, step)
                sess.run(tf.variables_initializer(local_vars_metrics))
                
            if step > 0 and step % (int(itr_per_epoch / FLAGS.valid_time)) == 0:
                sess.run(iterator_vld.initializer)
                sess.run(tf.variables_initializer(local_vars_metrics))
                TNR, F1, MCC, IoU, Recall, Prec, AUC = [], [], [], [], [], [], []
                warnings.simplefilter('ignore', RuntimeWarning)
                while True:
                    try:
                        labels_, preds_, _ = sess.run([labels_msk, preds_msk, update_metrics_count],
                                                      feed_dict={handle: handle_vld, is_training: False})
                        for i in range(labels_.shape[0]):
                            recall, tnr, prec, f1, mcc, iou, tn, tp, fn, fp = utils.metrics.get_metrics(labels_[i],
                                                                                                        preds_[i])
                            auc_score = roc_auc_score(labels_[i].reshape(-1,),preds_[i].reshape(-1,))
                            TNR.append(tnr)
                            F1.append(f1)
                            MCC.append(mcc)
                            IoU.append(iou)
                            Recall.append(recall)
                            Prec.append(prec)
                            AUC.append(auc_score)
                    except tf.errors.OutOfRangeError:
                        break
                mean_loss_, mean_accuracy_, summary_str = sess.run([mean_loss, mean_accuracy, summary_op])
                saver.save(sess, '{}/model.ckpt-{:g}-{:g}'.format(FLAGS.logdir, np.mean(F1),np.mean(AUC)), int(step / itr_per_epoch))
                if np.mean(F1) > best_metric:
                    best_metric = np.mean(F1)
                if np.mean(AUC)> best_auc_metric:
                    best_auc_metric = np.mean(AUC)
                if mean_loss_ < best_val_loss:
                    best_val_loss = mean_loss_
                    
                    loss_decrease = 0
                else:
                    loss_decrease += 1
                print_func(
                    'validation loss: {:g} ACC: {:g} Recall: {:g} Prec: {:g} TNR: {:g} F1: {:g} MCC: {:g} IoU: {:g} AUC: {:g} best_F1 metric: {:g} best AUC metric: {:g}'.format( \
                        mean_loss_, mean_accuracy_, np.mean(Recall), np.mean(Prec), np.mean(TNR), np.mean(F1),
                        np.mean(MCC), np.mean(IoU),np.mean(AUC), best_metric, best_auc_metric))
                print_func('loss_decreas: ', str(loss_decrease))
                summary_writer_vld.add_summary(summary_str, step)
                sess.run(tf.variables_initializer(local_vars_metrics))

                if (loss_decrease > 5):
                    break

    else:
        print_func('Mode not defined: ' + FLAGS.mode)
        return None


if __name__ == '__main__':
    tf.app.run()
