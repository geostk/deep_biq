import os

pwd = os.path.dirname(os.path.abspath(__file__))


def get_label(mos):
    label = -1
    if mos <= 20:
        label = 0
    elif mos <= 40:
        label = 1
    elif mos <= 60:
        label = 2
    elif mos <= 80:
        label = 3
    elif mos <= 100:
        label = 4
    return label


def gen_file(dir_name, out_file='train.txt'):
    with open(out_file, 'w') as f:
        mos = 0
        for f_name in os.listdir(dir_name):
            mos = float(f_name.split('_')[1])
            label = get_label(mos)
            f.write(os.path.join(dir_name, f_name) + ' ' + str(label) + '\n')


gen_file(os.path.join(pwd, 'rawdata/cropped_train'), 'quality_train.txt')
gen_file(os.path.join(pwd, 'rawdata/cropped_validation'), 'quality_validation.txt')
#gen_file(os.path.join(pwd, '/Users/andy/Downloads/cropped_train'), 'quality_train.txt')
#gen_file(os.path.join(pwd, '/Users/andy/Downloads/cropped_validation'), 'quality_validation.txt')
