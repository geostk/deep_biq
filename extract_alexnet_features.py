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
import numpy as np
import tensorflow as tf
from datetime import datetime
import time
from alexnet import AlexNet
from datagenerator import ImageDataGenerator

tf.app.flags.DEFINE_integer('batch_size', 64,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_classes', 2,
                            """Number of images to process in a batch.""")
FLAGS = tf.app.flags.FLAGS

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

# Path to the textfiles for the trainings and validation set
train_file = 'data/quality_train.txt'
val_file = 'data/quality_validation.txt'

# Learning params
learning_rate = 0.001
num_epochs = 1
batch_size = FLAGS.batch_size

# Network params
dropout_rate = 0.5
num_classes = FLAGS.num_classes
# train_layers = ['fc8', 'fc7']

# How often we want to write the tf.summary data to disk
display_step = 5

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "quality_training"
checkpoint_path = "alexnet_model/model_epoch10.ckpt"

# Create parent path if it doesn't exist

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
tf.summary.image('image', x, max_outputs=16)
y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, ['fc8'])  # don't load fc8

# Link variable to model output
features_op = model.fc8

# List of trainable variables of the layers we want to train
# var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]
var_list = [v for v in tf.trainable_variables()]

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Initalize the data generator seperately for the training and validation set
train_generator = ImageDataGenerator(train_file,
                                     horizontal_flip=True, shuffle=True)
val_generator = ImageDataGenerator(val_file, shuffle=False)

# Get the number of training/validation steps per epoch
train_batches_per_epoch = np.floor(train_generator.data_size / batch_size).astype(np.int16)
val_batches_per_epoch = np.floor(val_generator.data_size / batch_size).astype(np.int16)

# Start Tensorflow session
with tf.Session() as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)

    # Load the pretrained weights into the non-trainable layer
    # model.load_initial_weights(sess)
    saver.restore(sess, checkpoint_path)

    for _ in range(len(train_generator.images)):
        batch_tx, scores, paths = train_generator.next_batch_with_filename(2)
        print batch_tx, scores, paths
        features = sess.run(features_op, feed_dict={x: batch_tx,
                                                    keep_prob: 1.})
        print features

        step = step + 1
