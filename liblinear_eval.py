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
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
from scipy.stats import pearsonr

from data.image_processing import gen_boxes
from liblinearutil import load_model, predict
from alexnet import AlexNet

tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_classes', 5,
                            """Number of images to process in a batch.""")

tf.app.flags.DEFINE_string('checkpoint', 'alexnet_quality_model.tmp/model_epoch22.ckpt-0',
                           """Number of images to process in a batch.""")
FLAGS = tf.app.flags.FLAGS

validation_dir = 'data/rawdata/validation'
m = load_model("./liblinear.model")
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
saver.restore(sess, FLAGS.checkpoint)
# model.load_initial_weights(sess)

labels = []
preds_min = []
preds_avg = []
preds_max = []

def crop_a_image(image_tensor, box):
    image = tf.image.crop_to_bounding_box(image_tensor, box[0], box[1], box[2], box[3])
    return image
def crop_image(filename):
    with tf.gfile.FastGFile(filename, 'r') as ff:
        image_tensor = ff.read()
        _decode_jpeg = tf.image.decode_jpeg(image_tensor, channels=3)
        n_boxes = FLAGS.batch_size
        print (n_boxes)
        img = cv2.imread(filename)  # height*width*channels
        boxes = gen_boxes(img.shape[1], img.shape[0], 227, 227, n_boxes)
        result = []
        for i, box in enumerate(boxes):
            target_name = os.path.basename(filename.replace(".jpg", "_" + str(i) + ".jpg"))
            target_name = os.path.join('tmp', target_name)
            image = crop_a_image(_decode_jpeg, box)
            image = tf.squeeze(image)
            jpeg_bin = sess.run(image)
            Image.fromarray(np.asarray(jpeg_bin)).save(target_name)
            result.append(target_name)
        return result

mean = np.array([104., 117., 124.])
def read_one_img(path):
    img = cv2.imread(path)
    # flip image at random if flag is selected
    # rescale image
    # img = cv2.resize(img, (self.scale_size[0], self.scale_size[0]))
    img = img.astype(np.float32)

    # subtract mean
    img -= mean
    return img


def evaluate():
    for f_name in [os.path.join(validation_dir, f) for f in os.listdir(validation_dir)]:
        images_names = crop_image(f_name)
        images = np.ndarray([batch_size, 227, 227, 3])
        for i,image_name in enumerate(images_names):
            image=read_one_img(image_name)
            images[i]=image
        batch_tx =  images
        mos = float(f_name.split('_')[1].replace('.jpg', ''))
        features = sess.run(features_op, feed_dict={x: batch_tx, keep_prob: 1.})
        pred_score = predict([], features, m, options="")[0]
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
