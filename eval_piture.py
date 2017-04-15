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
from alexnet import AlexNet
from datagenerator import ImageDataGenerator
from image_processing import crop_a_image

tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_classes', 5,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('extracted_train_features_path', 'data/extracted_features.pl',
                           """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('extracted_validation_features_path', 'data/extracted_validation.pl',
                           """Number of images to process in a batch.""")
FLAGS = tf.app.flags.FLAGS

# Path to the textfiles for the trainings and validation set
train_file = 'data/quality_train.txt'
val_file = 'data/quality_validation.txt'

# Learning params
learning_rate = 0.001
num_epochs = 1
batch_size = FLAGS.batch_size

num_classes = FLAGS.num_classes
# train_layers = ['fc8', 'fc7']

# How often we want to write the tf.summary data to disk

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "quality_training"
checkpoint_path = "alexnet_model/model_epoch10.ckpt"

# Create parent path if it doesn't exist

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
train_generator = ImageDataGenerator(train_file,
                                     horizontal_flip=True, shuffle=True, nb_classes=num_classes)
val_generator = ImageDataGenerator(val_file, shuffle=False, nb_classes=num_classes)

# Get the number of training/validation steps per epoch
train_batches_per_epoch = np.floor(train_generator.data_size / batch_size).astype(np.int16)
val_batches_per_epoch = np.floor(val_generator.data_size / batch_size).astype(np.int16)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# saver.restore(sess, checkpoint_path)
model.load_initial_weights(sess)

validation_dir = 'data/rawdata/validation'

labels = []
preds_min = []
preds_avg = []
preds_max = []

def evaluate():
    for f_name in [os.path.join(validation_dir, f) for f in os.listdir(validation_dir)]:
        batch_tx = crop_a_image(f_name, 227, 227, FLAGS.batch_size)
        mos = float(f_name.split('_')[1].replace('.jpg', ''))
        print (mos)
        features = sess.run(features_op, feed_dict={x: batch_tx, keep_prob: 1.})
        labels.append(mos)


def main():
    evaluate()


if __name__ == '__main__':
    main()