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
tf.app.flags.DEFINE_string('check_point', 'alexnet_quality_model.tmp/model_epoch25.ckpt-0',
                           """Number of images to process in a batch.""")
FLAGS = tf.app.flags.FLAGS

# Path to the textfiles for the trainings and validation set
train_file = 'data/quality_train.txt'
val_file = 'data/quality_validation.txt'
num_classes = FLAGS.num_classes
# train_layers = ['fc8', 'fc7']

# How often we want to write the tf.summary data to disk

# Path for tf.summary.FileWriter and to store model checkpoints

# Create parent path if it doesn't exist

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [FLAGS.batch_size, 227, 227, 3])
y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, ['fc8'])  # don't load fc8

# Link variable to model output
features_op = model.fc7

saver = tf.train.Saver()

# Get the number of training/validation steps per epoch
sess = tf.Session()
saver.restore(sess, FLAGS.check_point)
print(FLAGS.check_point, "restored")


def export_to_liblinear(x_vals, y_vals, filename):
    print("generating liblinear features...")
    with open(filename, 'w') as f:
        for i, label in enumerate(y_vals):
            features = x_vals[i]
            line = str(label) + "\t"
            for k, v in enumerate(features):
                line = line + str(k + 1) + ":" + str(v) + " "
            line = line.strip()
            f.write(line + '\n')
    f.close()
    print("liblinear features done.")


def extract_one_image(f_name):
    batch_tx = crop_a_image(f_name, 227, 227, FLAGS.batch_size)
    features = sess.run(features_op, feed_dict={x: batch_tx, keep_prob: 1.})
    return features


def get_mos(f_name):
    mos = float(f_name.split('_')[1].replace('.jpg', ''))
    return mos


def extract(dir_name, plpath, liblinear_features_path):
    y_vals = np.array([])
    x_vals = np.ndarray(shape=[0, 4096])
    i = 1
    for f_name in [os.path.join(dir_name, f) for f in os.listdir(dir_name)]:
        mos = get_mos(f_name)
        scores = [mos for i in range(FLAGS.batch_size)]
        features = extract_one_image(f_name)
        x_vals = np.append(x_vals, features, axis=0)
        y_vals = np.append(y_vals, scores)
        if i % 100 == 0:
            with open(plpath, 'w') as  f:
                features_map = {}
                features_map['x'] = x_vals
                features_map['y'] = y_vals
                pickle.dump(features_map, f)
            export_to_liblinear(x_vals, y_vals, liblinear_features_path)


def main():
    extract('data/rawdata/train', 'data/22_train.pl', 'data/22_train.features.txt')
    extract('data/rawdata/validation', 'data/22_validation.pl', 'data/22_valid.features.txt')


if __name__ == '__main__':
    main()
