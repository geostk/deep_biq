import numpy as np
from scipy.stats import pearsonr

exclude_list = ['solver_type', 'nr_class', 'nr_feature', 'bias', 'w']
modle_file = 'liblinear.model'

model = []


def black_line(line):
    for i in exclude_list:
        if line.startswith(i):
            return True
    return False


with open(modle_file) as f:
    i = 0
    for line in f:
        line = line.strip()
        if black_line(line):
            continue
        model.append(float(line))
        i += 1

def score(xs):
    return np.matmul(xs, model)
print model

