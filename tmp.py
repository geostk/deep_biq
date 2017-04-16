import random

import numpy as np
import os
from scipy.stats import pearsonr
import cPickle as pickle

feature_dir = 'data/features'


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


def main():
    export_to_liblinear_format()


if __name__ == '__main__':
    main()
