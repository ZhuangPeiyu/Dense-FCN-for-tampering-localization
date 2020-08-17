import os
import tensorflow as tf
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('filename', 'train.py', 'the file to run (train.py or test.py)')

os.system("python3 ./"+FLAGS.filename)
