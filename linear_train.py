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

from scipy.stats import pearsonr

from linear_alexnet import AlexNet
from linear_datagenerator import ImageDataGenerator

tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_classes', 5,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.8, 'decay factor')
tf.app.flags.DEFINE_float('initial_learning_rate', 0.0005, 'init learning rate')
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
train_file = 'data/quality_linear_train.txt'
val_file = 'data/quality_linear_validation.txt'

# Learning params
# learning_rate = 0.001
num_epochs = 5000
batch_size = FLAGS.batch_size

# Network params
dropout_rate = 0.5
num_classes = FLAGS.num_classes
train_layers = ['fc9']

# How often we want to write the tf.summary data to disk
display_step = 10

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "linear_quality_training"
checkpoint_path = "linear_alexnet_quality_model"
fine_tuned_model_path = 'alexnet_quality_model'
# Initalize the data generator seperately for the training and validation set
train_generator = ImageDataGenerator(train_file, shuffle=True, nb_classes=num_classes)
val_generator = ImageDataGenerator(val_file, shuffle=False, nb_classes=num_classes)
# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
tf.summary.image('image', x, max_outputs=16)
y = tf.placeholder(tf.float32, [None])
keep_prob = tf.placeholder(tf.float32)
# Initialize model
model = AlexNet(x, keep_prob, num_classes, ['fc8'])  # don't load fc8
score_op = model.fc9


# Link variable to model output
def loss(pred, value):
    with tf.name_scope("l1_loss"):
        l1_loss = tf.reduce_mean(tf.abs(tf.subtract(pred, value)))
    tf.summary.scalar('l1_loss', l1_loss)
    with tf.name_scope('regularize_loss'):
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    tf.summary.histogram('regularization_losse', regularization_losses)
    total_loss = l1_loss + 0.01 * sum(regularization_losses)
    loss_averages = tf.train.ExponentialMovingAverage(0.9)
    loss_averages_op = loss_averages.apply([l1_loss] + [total_loss])
    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)
    return total_loss


# List of trainable variables of the layers we want to train
# var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]
restore_vars = [v for v in tf.trainable_variables() if v.name.split('/')[0] not in ['fc9']]
# var_list = [v for v in tf.trainable_variables()]
val_batches_per_epoch = np.floor(val_generator.data_size / batch_size).astype(np.int16)
global_step = tf.get_variable('global_step', [],
                              initializer=tf.constant_initializer(0), trainable=False)
decay_steps = val_batches_per_epoch * 100
print (decay_steps)
learning_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                           global_step,
                                           decay_steps,
                                           FLAGS.learning_rate_decay_factor,
                                           staircase=True)
with tf.name_scope('loss'):
    loss = loss(score_op, y)

# Train op
with tf.name_scope("train"):
    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)
    tf.summary.scalar('learning_rate', learning_rate)
# Add gradients to summary
# Add the loss to summary
tf.summary.scalar('total_loss', loss)

# Evaluation op: Accuracy of the model
# with tf.name_scope("accuracy"):
#    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
#    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Add the accuracy to the summary
lcc_op = tf.placeholder(dtype=tf.float32, name='lcc')
tf.summary.scalar('lcc', lcc_op)
for var in tf.trainable_variables():
    print('aaa', var.name)
    tf.summary.histogram(var.name, var)
# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver(restore_vars)

# Get the number of training/validation steps per epoch
train_batches_per_epoch = np.floor(train_generator.data_size / batch_size).astype(np.int16)

# Start Tensorflow session
with tf.Session() as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)
    # Decay the learning rate exponentially based on the number of steps.
    # Load the pretrained weights into the non-trainable layer
    model.load_initial_weights(sess)
    # /saver.restore(sess, os.path.join(fine_tuned_model_path, 'model_epoch23.ckpt-0'))
    saver = tf.train.Saver(max_to_keep=100)
    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))
    # Loop over number of epochs
    for epoch in range(num_epochs):

        print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

        step = 1

        lcc = 0.
        while step < train_batches_per_epoch:
            start_time = time.time()
            # Get a batch of images and labels
            batch_xs, batch_ys = train_generator.next_batch(batch_size)

            # And run the training op
            _, loss_value = sess.run([train_op, loss], feed_dict={x: batch_xs,
                                                                  y: batch_ys,
                                                                  keep_prob: 1.0})
            duration = time.time() - start_time
            # Generate summary with the current batch of data and write to file
            if step % display_step == 0:
                examples_per_sec = FLAGS.batch_size / float(duration)
                format_str = ('%s: step %d, loss = %.2f, lcc = %.4f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value, lcc,
                                    examples_per_sec, duration))

                s = sess.run(merged_summary, feed_dict={x: batch_xs,
                                                        y: batch_ys,
                                                        keep_prob: 1., lcc_op: lcc})

                writer.add_summary(s, epoch * train_batches_per_epoch + step)
            if step % 100 == 0:
                print("{} Saving checkpoint of model...".format(datetime.now()))
                # save checkpoint of the model
                checkpoint_name = os.path.join(checkpoint_path, 'model_epoch' + str(epoch + 1) + '.ckpt')
                save_path = saver.save(sess, checkpoint_name, global_step=global_step)
                print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))

            step += 1

        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))
        test_acc = 0.
        test_count = 1
        tys = []
        scores = []
        for _ in range(val_batches_per_epoch):
            batch_tx, batch_ty = val_generator.next_batch(batch_size)
            score = sess.run(score_op, feed_dict={x: batch_tx, y: batch_ty, keep_prob: 1., lcc_op: lcc})
            tys.extend(batch_ty)
            scores.extend(np.squeeze(score))

        lcc = pearsonr(tys, scores)[0]
        print("{} Validation lcc = {}".format(datetime.now(), lcc))

        # Reset the file pointer of the image data generator
        val_generator.reset_pointer()
        train_generator.reset_pointer()
        print("{} Saving checkpoint of model...".format(datetime.now()))
        # save checkpoint of the model
        checkpoint_name = os.path.join(checkpoint_path, 'model_epoch' + str(epoch + 1) + '.ckpt')
        save_path = saver.save(sess, checkpoint_name, global_step=global_step)

        print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
