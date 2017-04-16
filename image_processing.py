import random

import cv2, os
import numpy as np

mean = np.array([104., 117., 124.])


def gen_boxes(org_width, org_height, target_width, target_height, num=30):
    width_offsets = [random.randint(0, org_width - target_width) for i in xrange(num)]
    height_offsets = [random.randint(0, org_height - target_height) for i in xrange(num)]
    result = []
    for i in range(0, num):
        result.append((height_offsets[i], width_offsets[i], target_height, target_width))
    return result


def crop_a_image(image_path, target_width, target_height, num=30):
    img = cv2.imread(image_path)  # height*width*channels
    img = img.astype(np.float32)
    boxes = gen_boxes(img.shape[1], img.shape[0], target_width, target_height, num)
    images = np.ndarray([num, target_height, target_width, 3])
    for i, box in enumerate(boxes):
        image = img[box[0]:box[0] + target_height, box[1]:box[1] + target_width]
        #image -= mean
        images[i] = image
    return images
