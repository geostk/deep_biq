import cPickle as pickle
import os
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR, LinearSVR

pls_dir = 'data/features/train/'
y_vals = np.array([])
x_vals = np.ndarray(shape=[0, 4096])
svr_lin = SVR(kernel='rbf', C=1e3)
for file_name in os.listdir(pls_dir):
    if not file_name.endswith('.pl'):
        continue
    with open(os.path.join(pls_dir, file_name)) as f:
        f_map = pickle.load(f)
        y = f_map.get("y")
        x = f_map.get("x")
        x_vals = np.append(x_vals, x, axis=0)
        y_vals = np.append(y_vals, y)
        print (file_name, "loaded")
shuffle_indexes = np.random.choice(len(y_vals), len(y_vals), replace=False)
y_vals_train = y_vals[shuffle_indexes]
x_vals_train = x_vals[shuffle_indexes]
print y_vals_train.shape
print x_vals_train.shape
# linear_svr = LinearSVR(C=1e3)
linear_svr = SGDRegressor()
linear_svr.fit(x_vals_train, y_vals_train)
score_vals = np.array([])
label_vals = np.array([])
for i in range(1000):
    rand_index = np.random.choice(len(x_vals_train), size=128)
    y_train = y_vals_train[rand_index]
    x_train = x_vals_train[rand_index]
    y = linear_svr.predict(x_train)
    score_vals = np.append(score_vals, y)
    label_vals = np.append(label_vals, y_train)
    print pearsonr(y, y_train)[0]
