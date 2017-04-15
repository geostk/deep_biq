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
tf.app.flags.DEFINE_integer('num_classes', 5,
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
num_epochs = 5000
batch_size = FLAGS.batch_size

# Network params
dropout_rate = 0.5
num_classes = FLAGS.num_classes
# train_layers = ['fc8', 'fc7']

# How often we want to write the tf.summary data to disk
display_step = 5

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "quality_training"
checkpoint_path = "alexnet_quality_model"

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
tf.summary.image('image', x, max_outputs=16)
y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, ['fc8'])  # don't load fc8

# Link variable to model output
score = model.fc8

# List of trainable variables of the layers we want to train
# var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]
var_list = [v for v in tf.trainable_variables()]

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=y))

# Train op
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# Add gradients to summary
for gradient, var in gradients:
    tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary
for var in var_list:
    tf.summary.histogram(var.name, var)

# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)

# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Add the accuracy to the summary
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Initalize the data generator seperately for the training and validation set
train_generator = ImageDataGenerator(train_file, shuffle=True, nb_classes=num_classes)
val_generator = ImageDataGenerator(val_file, shuffle=False, nb_classes=num_classes)

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
    #model.load_initial_weights(sess)
    saver.restore(sess, os.path.join(checkpoint_path,'model_epoch4.ckpt'))

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))

    # Loop over number of epochs
    for epoch in range(num_epochs):

        print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

        step = 1

        while step < train_batches_per_epoch:
            start_time = time.time()
            # Get a batch of images and labels
            batch_xs, batch_ys = train_generator.next_batch(batch_size)

            # And run the training op
            _, loss_value = sess.run([train_op, loss], feed_dict={x: batch_xs,
                                                                  y: batch_ys,
                                                                  keep_prob: dropout_rate})
            duration = time.time() - start_time
            # Generate summary with the current batch of data and write to file
            if step % display_step == 0:
                examples_per_sec = FLAGS.batch_size / float(duration)
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, duration))

                s = sess.run(merged_summary, feed_dict={x: batch_xs,
                                                        y: batch_ys,
                                                        keep_prob: 1.})
                writer.add_summary(s, epoch * train_batches_per_epoch + step)
            if step % 100 == 0:
                print("{} Saving checkpoint of model...".format(datetime.now()))

                # save checkpoint of the model
                checkpoint_name = os.path.join(checkpoint_path, 'model_epoch' + str(epoch + 1) + '.ckpt')
                save_path = saver.save(sess, checkpoint_name)

                print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))

            step += 1

        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))
        test_acc = 0.
        test_count = 1
        for _ in range(val_batches_per_epoch):
            batch_tx, batch_ty = val_generator.next_batch(batch_size)
            acc = sess.run(accuracy, feed_dict={x: batch_tx,
                                                y: batch_ty,
                                                keep_prob: 1.})
            test_acc += acc
            test_count += 1
        test_acc /= test_count
        print("Validation Accuracy = {:.4f}".format(datetime.now(), test_acc))

        # Reset the file pointer of the image data generator
        #val_generator.reset_pointer()
        #train_generator.reset_pointer()
        print("{} Saving checkpoint of model...".format(datetime.now()))

        # save checkpoint of the model
        checkpoint_name = os.path.join(checkpoint_path, 'model_epoch' + str(epoch + 1) + '.ckpt')
        save_path = saver.save(sess, checkpoint_name)

        print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
