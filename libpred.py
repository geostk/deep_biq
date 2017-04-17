import numpy as np
from scipy.stats import pearsonr
import os

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


def evaluate():
    evaluate_dir = 'data/features/validation'
    files = [os.path.join(evaluate_dir, f) for f in os.listdir(evaluate_dir)]

    labels = []
    avg_scores = []
    max_scores = []
    min_scores = []

    for f_name in files:
        if not f_name.endswith('.txt'): continue
        preds = []
        with open(f_name) as f:
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
                preds.append(score(xs))

            labels.append(float(label))
            avg_scores.append(np.average(preds))
            max_scores.append(np.min(preds))
            min_scores.append(np.max(preds))
            print(float(label), np.average(preds), np.min(preds), np.max(preds))
            print(float(label), pearsonr(labels, avg_scores), pearsonr(labels, min_scores), pearsonr(labels, max_scores))


def main():
    evaluate()
    # test_liblinear()
    # evaluate()


if __name__ == '__main__':
    main()
