import StringIO
import os

import cv2
from PIL import Image

import numpy as np
import random

from image_processing import gen_boxes, crop_a_image

ORIG_WIDTH = 500
ORIG_HEIGHT = 500

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


def crop_a_image(image_path, target_width, target_height, num=30):
    img = cv2.imread(image_path)  # height*width*channels
    img = img.astype(np.float32)
    boxes = gen_boxes(img.shape[1], img.shape[0], target_width, target_height, num)
    images = np.ndarray([num, target_height, target_width, 3])
    for i, box in enumerate(boxes):
        image = img[box[0]:box[0] + target_height, box[1]:box[1] + target_width]
        # image -= mean
        images[i] = image
    return images


def crop_images():
    for filename in files:
        target_name = filename.replace("train", "cropped_train")
        target_name = target_name.replace("validation", "cropped_validation")
        print(filename)
        images = crop_a_image(filename, 227, 227, get_boxes_number(filename))
        for i, image in enumerate(images):
            save_name = target_name.replace(".jpg", "_" + str(i) + ".jpg")

            print(save_name)
            cv2.imwrite(save_name, image)


def main():
    crop_images()


if __name__ == '__main__':
    main()
