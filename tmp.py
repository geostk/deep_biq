import random
from sklearn import datasets

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


(x_vals, y_vals) = datasets.make_regression(n_samples=1000, n_features=10, noise=5)
train_indices = np.random.choice(len(x_vals), int(round(len(x_vals) * 0.8)), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
y_vals_train = y_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_test = y_vals[test_indices]

def main():
    export_to_liblinear(x_vals_train,y_vals_train,'train.txt')
    export_to_liblinear(x_vals_test,y_vals_test,'valid.txt')


if __name__ == '__main__':
    main()
