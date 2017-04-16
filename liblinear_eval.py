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
import random
import threading
from Queue import Queue

import numpy as np
from scipy.stats import pearsonr
import cPickle as pickle

from extract_alexnet_features import extract_one_image, get_mos, feature_dir, export_to_liblinear
from liblinearutil import load_model, predict

validation_dir = 'data/rawdata/validation'
m = load_model("./liblinear.model")

labels = []
preds_min = []
preds_avg = []
preds_max = []


def evaluate():
    for f_name in [os.path.join(validation_dir, f) for f in os.listdir(validation_dir)]:
        features = extract_one_image(f_name)
        pred_score = predict([], features, m, options="")[0]
        preds_min.append(np.min(pred_score))
        preds_max.append(np.max(pred_score))
        preds_avg.append(np.average(pred_score))
        mos = get_mos(f_name)
        print (mos, np.average(pred_score), np.max(pred_score), np.min(pred_score))
        labels.append(mos)
        print("avg_lcc", pearsonr(labels, preds_avg)[0], "min_lcc", pearsonr(labels, preds_min)[0], "max_lcc",
              pearsonr(labels, preds_max)[0])


q = Queue()
y_vals = np.array([])
x_vals = np.ndarray(shape=[0, 4096])


def work():
    file = q.get()
    with open(file) as f:
        print file
        features_map = pickle.load(f)


def export_to_liblinear_format():
    print('loading data')
    y_vals = np.array([])
    x_vals = np.ndarray(shape=[0, 4096])
    train_dir = os.path.join(feature_dir, 'train.tmp')
    files = [os.path.join(train_dir, f_name) for f_name in os.listdir(train_dir)]
    random.shuffle(files)
    for file in files:
        if not file.endswith('.pl'):
            continue
        print(file, ' loaded')
        with open(file) as f:
            features_map = pickle.load(f)
            x = features_map.get('x')
            y = features_map.get('y')
            x_vals = np.append(x_vals, x, axis=0)
            y_vals = np.append(y_vals, y)
    shuffle_indexes = np.random.choice(len(y_vals), len(y_vals), replace=False)
    y_vals = y_vals[shuffle_indexes]
    x_vals = x_vals[shuffle_indexes]
    export_to_liblinear(x_vals, y_vals, 'data/25.features.txt')
    # pred_score = predict([], x_vals, m, options="")[0]
    # for i, score in enumerate(pred_score):
    #    print(score, y_vals[i])


def main():
    train_dir = os.path.join(feature_dir, 'train.tmp')
    files = [os.path.join(train_dir, f_name) for f_name in os.listdir(train_dir)]
    for file in files:
        if not file.endswith('.pl'):
            continue
        q.put(file)
    for i in range(100):

        t = threading.Thread(target=work)
        t.daemon = True
        t.start()
    q.join()

    #test_liblinear()
    # evaluate()


if __name__ == '__main__':
    main()
