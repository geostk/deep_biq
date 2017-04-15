import os


def gen_file(dir_name, out_file='train.txt'):
    with open(out_file, 'w') as f:
        label = 0
        for f_name in os.listdir(dir_name):
            if f_name.startswith('dog'):
                label = 1
            elif f_name.startswith('cat'):
                label = 0
            f.write(os.path.join(dir_name, f_name) + ' ' + str(label) + '\n')


gen_file('/Users/andy/Downloads/test', 'valid.txt')
gen_file('/Users/andy/Downloads/train', 'train.txt')
