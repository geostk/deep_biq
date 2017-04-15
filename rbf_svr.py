import cPickle as pickle
import os
import numpy as np
from sklearn.svm import SVR

features_map = {}
data_file = 'data/original_train.pl'
with open(data_file) as f:
    features_map = pickle.load(f)
x_vals = features_map.get('x')
y_vals = features_map.get('y')
shuffle_indexes = np.random.choice(len(y_vals), len(y_vals), replace=False)
y_vals_train = y_vals[shuffle_indexes]
x_vals_train = x_vals[shuffle_indexes]
print y_vals.shape
svr_lin = SVR(kernel='rbf', C=1e3)
for i in range(1000):
    print (i)
    rand_index = np.random.choice(len(x_vals_train), size=4096)
    y_train = y_vals_train[rand_index]
    x_train = x_vals_train[rand_index]
    svr_lin.fit(x_train, y_train)
    if i % 100 == 0:
        with open('svr_model', 'w') as  f:
            pickle.dump(svr_lin, f)
