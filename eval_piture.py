"""
With this script you can finetune AlexNet as provided in the alexnet.py
class on any given dataset. 
Specify the configuration settings at the beginning according to your 
problem.
This script was written for TensorFlow 1.0 and come with a blog post 
you can find here:

https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

Author: Frederik Kratzert 
contact: f.kratzert(at)gmail.com
"""
import os
import cPickle as pickle
import numpy as np
import tensorflow as tf
from scipy.stats import pearsonr

from alexnet import AlexNet
from datagenerator import ImageDataGenerator
from image_processing import crop_a_image
from libpred import score

tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_classes', 5,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('extracted_train_features_path', 'data/extracted_features.pl',
                           """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('extracted_validation_features_path', 'data/extracted_validation.pl',
                           """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('checkpoint', 'alexnet_quality_model.tmp/model_epoch17.ckpt-0',
                           """Number of images to process in a batch.""")
FLAGS = tf.app.flags.FLAGS

# Path to the textfiles for the trainings and validation set

# Learning params
learning_rate = 0.001
num_epochs = 1
batch_size = FLAGS.batch_size

num_classes = FLAGS.num_classes
# train_layers = ['fc8', 'fc7']

# How often we want to write the tf.summary data to disk

# Path for tf.summary.FileWriter and to store model checkpoints

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, ['fc8'])  # don't load fc8

# Link variable to model output
features_op = model.fc7

# List of trainable variables of the layers we want to train
# var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]
# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Initalize the data generator seperately for the training and validation set
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# saver.restore(sess, checkpoint_path)
# model.load_initial_weights(sess)

validation_dir = 'data/rawdata/validation'

labels = []
preds_min = []
preds_avg = []
preds_max = []
with open('svr_model') as f:
    svr_lin = pickle.load(f)


def evaluate():
    for f_name in [os.path.join(validation_dir, f) for f in os.listdir(validation_dir)]:
        batch_tx = crop_a_image(f_name, 227, 227, FLAGS.batch_size)
        mos = float(f_name.split('_')[1].replace('.jpg', ''))
        features = sess.run(features_op, feed_dict={x: batch_tx, keep_prob: 1.})
        # pred_score = svr_lin.predict(features)
        pred_score = score(features)
        preds_min.append(np.min(pred_score))
        preds_max.append(np.max(pred_score))
        preds_avg.append(np.average(pred_score))
        print (mos, np.average(pred_score), np.max(pred_score), np.min(pred_score))
        labels.append(mos)
        print("avg_lcc", pearsonr(labels, preds_avg)[0], "min_lcc", pearsonr(labels, preds_min)[0], "max_lcc",
              pearsonr(labels, preds_max)[0])


def main():
    evaluate()


if __name__ == '__main__':
    main()
