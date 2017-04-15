import cPickle as pickle
import os
import numpy as np
from sklearn.svm import SVR

from image_processing import get_weigh


import cPickle as pickle
import os
import numpy as np
from sklearn.svm import SVR

from image_processing import get_weigh


def export_to_liblinear(x_vals, y_vals):
    with open('featues.txt', 'w') as f:
        for i, label in enumerate(y_vals):
            features = x_vals[i]
            line = str(label) + "\t"
            for k, v in enumerate(features):
                line = line + str(k + 1) + ":" + str(v) + " "
            line = line.strip()
            f.write(line + '\n')
    f.close()


train_file_dir = 'data/rawdata/trainpl/'
files = os.listdir(train_file_dir)
y_vals = np.array([])
x_vals = np.ndarray(shape=[0, 2048])
svr_lin = SVR(kernel='rbf', C=1e3)
for file_name in files:
    with open(os.path.join(train_file_dir, file_name)) as f:
        f_map = pickle.load(f)
        y = f_map.get("y")
        x = f_map.get("x")
        x_vals = np.append(x_vals, x, axis=0)
        y_vals = np.append(y_vals, y)
        print (file_name, "loaded")


weighs = np.array([get_weigh(y) for y in y_vals])
shuffle_indexes = np.random.choice(len(y_vals), len(y_vals), replace=False)
y_vals_train = y_vals[shuffle_indexes]
x_vals_train = x_vals[shuffle_indexes]
weigh_train = weighs[shuffle_indexes]
# export_to_liblinear(x_vals_train, y_vals_train)
print y_vals.shape
for i in range(10000000):
    print (i)
    rand_index = np.random.choice(len(x_vals_train), size=4096)
    y_train = y_vals_train[rand_index]
    x_train = x_vals_train[rand_index]
    w_train = weigh_train[rand_index]
    svr_lin.fit(x_train, y_train, sample_weight=w_train)
    if i % 10 == 0:
        with open('rbf_svr_model', 'w') as  f:
            pickle.dump(svr_lin, f)
