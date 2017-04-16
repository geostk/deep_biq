import numpy as np
from scipy.stats import pearsonr

exclude_list = ['solver_type', 'nr_class', 'nr_feature', 'bias', 'w']
modle_file = 'liblinear.model'

weigh = []


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
        weigh.append(float(line))
        i += 1
print (len(weigh), weigh)


def score(xs):
    result = 0
    for i in range(0, 4096):
        result = result + weigh[i] * xs[i]
    return result


y1s = []
y2s = []
with open('1000.txt') as f:
    for line in f:
        line = line.strip()
        label_and_feature = line.split('\t')
        label = label_and_feature[0]
        features = label_and_feature[1]
        xs = list()
        try:
            for feature in features.split(' '):
                xi = feature.split(':')[1]
                xs.append(float(xi))
        except:
            print features
            break

        print label, score(xs)
        y1s.append(float(label))
        y2s.append(score(xs))
        print(pearsonr(y1s, y2s))
