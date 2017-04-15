import StringIO
import os

from PIL import Image

import numpy as np
import tensorflow as tf
import random

FLAGS = tf.app.flags.FLAGS
from image_processing import gen_boxes, crop_a_image

ORIG_WIDTH = 500
ORIG_HEIGHT = 500
TARGET_WIDTH = FLAGS.image_size
TARGET_HEIGHT = FLAGS.image_size


def convert_to_draw_boxes(boxes, width, height):
    return [(box[0] / height, box[1] / width, (box[0] + box[2]) / height, (box[1] + box[3]) / width) for box in boxes]


NUM_BOXES = 30


def get_boxes_number(file_name):
    l = file_name.split("_")[1].split(".")[0]
    label = float(l.strip())
    weight = 0
    if label <= 20:
        weight = 5.35
    elif label <= 40:
        weight = 2.60
    elif label <= 60:
        weight = 1.51
    elif label <= 80:
        weight = 1.0
    elif label <= 100:
        weight = 6.4
    return int(weight * NUM_BOXES)


sess = tf.Session()
# boxes_for_draw = convert_to_draw_boxes(boxes, ORIG_WIDTH, ORIG_HEIGHT)
valid_dir = "rawdata/validation"
train_dir = "rawdata/train"
cropped_train_dir = "rawdata/cropped_train"
cropped_validation_dir = "rawdata/cropped_validation"
if not os.path.exists(cropped_train_dir): os.mkdir(cropped_train_dir)
if not os.path.exists(cropped_validation_dir): os.mkdir(cropped_validation_dir)

files = [os.path.join(train_dir, x) for x in os.listdir(train_dir)]
files.extend([os.path.join(valid_dir, x) for x in os.listdir(valid_dir)])
random.shuffle(files)
for filename in files:
    with tf.gfile.FastGFile(filename, 'r') as ff:
        image_tensor = ff.read()
        _decode_jpeg = tf.image.decode_jpeg(image_tensor, channels=3)
        n_boxes = get_boxes_number(filename)
        print (n_boxes)
        boxes = gen_boxes(ORIG_WIDTH, ORIG_HEIGHT, TARGET_WIDTH, TARGET_WIDTH, n_boxes)
        print (len(boxes), "ffff")
        filename = filename.replace("train", "cropped_train")
        filename = filename.replace("validation", "cropped_validation")
        for i, box in enumerate(boxes):
            target_name = filename.replace(".jpg", "_" + str(i) + ".jpg")
            print(target_name)
            if os.path.exists(target_name):
                print(target_name, "passed")
                continue
            image = crop_a_image(_decode_jpeg, box)
            image = tf.squeeze(image)
            jpeg_bin = sess.run(image)
            Image.fromarray(np.asarray(jpeg_bin)).save(target_name)
    ff.close()








# Generate a single distorted bounding box.
# jbegin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
#    tf.shape(image),
#    bounding_boxes=bounding_boxes)
