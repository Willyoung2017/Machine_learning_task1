import numpy as np
import pandas as pd
import pickle
from scipy import sparse
from scipy.sparse import hstack,vstack,csc_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

num_of_lines = 321910
test_num_of_lines = 40239
processed_text_path = "E:\\Machine learning\\Task1\\source\\processed_texts.pkl"
processed_title_path = "E:\\Machine learning\\Task1\\source\\processed_titles.pkl"
test_processed_text_path = "E:\\Machine learning\\Task1\\source\\test_processed_texts.pkl"
test_processed_title_path = "E:\\Machine learning\\Task1\\source\\test_processed_titles.pkl"
train_y_path = "E:\\Machine learning\\Task1\\train_data\\train.csv"
test_y_path = "E:\\Machine learning\\Task1\\test_data\\sample_submission.csv"


def load_data(path, num):
    ifile = open(path, 'rb')
    processed_content = list(range(num))
    for i in range(num):
        line = pickle.load(ifile)
        line = " ".join(line)
        processed_content[i] = line
    ifile.close()
    return processed_content

def dump_data(path, data):
    ofile = open(path, 'wb')
    pickle.dump(data, ofile)
    ofile.close()

print('load data...\n')

# load x
'''
ifile = open(processed_text_path, 'rb')
ifile_title = open(processed_title_path, 'rb')
test_ifile = open(test_processed_text_path, 'rb')
test_ifile_title = open(test_processed_title_path, 'rb')
processed_text = list(range(num_of_lines))
test_processed_text = list(range(test_num_of_lines))
for i in range(num_of_lines):
    line = pickle.load(ifile)
    line = " ".join(line)
    processed_text[i] = line
ifile.close()
for i in range(test_num_of_lines):
    line = pickle.load(test_ifile)
    line = " ".join(line)
    test_processed_text[i] = line
test_ifile.close()
'''
train_samplex = load_data(processed_text_path, num_of_lines)
train_samplex_title = load_data(processed_title_path, num_of_lines)
test_samplex = load_data(test_processed_text_path, test_num_of_lines)
test_samplex_title = load_data(test_processed_title_path, test_num_of_lines)
print('loading x succeed.\n')

# load y
train_sampley = pd.read_csv(train_y_path)
y_train = train_sampley['pred']

print('loading y succeed.\n')

# vectorizing
print('vectorizing data...\n')
maxfeature = 1000000
count_vec = TfidfVectorizer(binary = False, decode_error = 'ignore', max_features = maxfeature)
x_train_content = count_vec.fit_transform(train_samplex)
x_test_content = count_vec.transform(test_samplex)
x_train_title = 0.8 * count_vec.fit_transform(train_samplex_title)
x_test_title = 0.8 * count_vec.transform(test_samplex_title)
x_train = hstack((x_train_content, x_train_title))
x_test = hstack((x_test_content, x_test_title))
print('vectorizing finished.\n')

# dump data
print('dumping data...\n')
path_test_x = "E:\\Machine learning\\Task1\\source\\test_x.pkl"
path_train_x = "E:\\Machine learning\\Task1\\source\\train_x.pkl"
path_train_y = "E:\\Machine learning\\Task1\\source\\train_y.pkl"


dump_data(path_train_x, x_train)
print('train_x dumping finished...\n')
dump_data(path_train_y, y_train)
print('train_y dumping finished...\n')
dump_data(path_test_x, x_test)
print('test_x dumping finished...\n')

print('dumping finished.\n')