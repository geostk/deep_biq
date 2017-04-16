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
from extract_alexnet_features import extract_one_image, get_mos
from image_processing import crop_a_image
from libpred import score


validation_dir = 'data/rawdata/train'

labels = []
preds_min = []
preds_avg = []
preds_max = []
with open('rbf_svr_model') as f:
    svr_lin = pickle.load(f)


def evaluate():
    for f_name in [os.path.join(validation_dir, f) for f in os.listdir(validation_dir)]:
        features = extract_one_image(f_name)
        pred_score = svr_lin.predict(features)
        mos = get_mos(f_name)
        #print features.shape
        # pred_score = score(features)
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
