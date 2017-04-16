import os

pwd = os.path.dirname(os.path.abspath(__file__))





def gen_file(dir_name, out_file='train.txt'):
    with open(out_file, 'w') as f:
        for f_name in os.listdir(dir_name):
            mos = float(f_name.split('_')[1])
            f.write(os.path.join(dir_name, f_name) + ' ' + str(int(mos)) + '\n')


#gen_file(os.path.join(pwd, 'rawdata/cropped_train'), 'quality_linear_train.txt')
#gen_file(os.path.join(pwd, 'rawdata/cropped_validation'), 'quality_linear_validation.txt')
gen_file(os.path.join(pwd, '/Users/andy/Downloads/cropped_train'), 'quality_linear_train.txt')
gen_file(os.path.join(pwd, '/Users/andy/Downloads/cropped_validation'), 'quality_linear_validation.txt')
