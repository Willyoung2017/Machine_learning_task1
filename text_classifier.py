import json
import re
from bs4 import BeautifulSoup
import jieba
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack,vstack,csc_matrix

p_train = 1
p_test = 0
p_x = 1
p_y = 0
num_of_train = 321910
num_of_test = 40239
json_path_test = "E:\\Machine learning\\Task1\\test_data\\test.json"
json_path_train = "E:\\Machine learning\\Task1\\train_data\\train.json"
write_path_test_title = "E:\\Machine learning\\Task1\\source\\test_cut_titles.pkl"
write_path_test_content = "E:\\Machine learning\\Task1\\source\\test_cut_texts.pkl"
write_path_train_title = "E:\\Machine learning\\Task1\\source\\cut_titles.pkl"
write_path_train_content = "E:\\Machine learning\\Task1\\source\\cut_texts.pkl"
stop_word_path = "E:\\Machine learning\\Task1\\source\\stop_word.txt"
train_y_path = "E:\\Machine learning\\Task1\\train_data\\train.csv"
test_y_path = "E:\\Machine learning\\Task1\\test_data\\sample_submission.csv"
save_path_test_x = "E:\\Machine learning\\Task1\\source\\test_x.pkl"
save_path_train_x = "E:\\Machine learning\\Task1\\source\\train_x.pkl"
save_path_train_y = "E:\\Machine learning\\Task1\\source\\train_y.pkl"
submitting_path = "E:\\Machine learning\\Task1\\submission\\submission1_logistic_regression.csv"
with open(stop_word_path, 'r', encoding='UTF-8') as f_stop_word:
    stop_word_list = [line.rstrip() for line in f_stop_word]

def remove_words(_line):
    # deduplication
    line = list(set(_line))
    line_tmp = line.copy()
    # remove stop_words & numbers
    for word in line_tmp:
        if word in stop_word_list:
            line.remove(word)
        elif word[0] >= '0' and word[0] <= '9':
            line.remove(word)
    return line

def load_file(path):
    ifile = open(path, 'rb')
    data = pickle.load(ifile)
    ifile.close()
    return data


def load_data(path, train_or_test, x_or_y):
    if train_or_test == 1:
        num = num_of_train
    else:
        num = num_of_test
    if x_or_y == 1:
        with open(path, 'rb') as ifile:
            sample = []
            for i in range(num):
                line = pickle.load(ifile)
                sample.append(line)
                if(int((i * 100)/num) > int((i*100-100)/num)):
                    print("loading %d%%ok!\n" % (i * 100 / num))
    else:
        sample_y = pd.read_csv(path)
        sample = sample_y['pred']

    return sample


def jieba_cut(train_or_test):
    cnt = 0
    if train_or_test == 1:
        print("jieba_cut for train data start.\n")
        offile_for_title = open(write_path_train_title, 'wb')
        offile_for_content = open(write_path_train_content, 'wb')
        read_path = json_path_train
        num = num_of_train
    else:
        print("jieba_cut for test data start.\n")
        offile_for_title = open(write_path_test_title, 'wb')
        offile_for_content = open(write_path_test_content, 'wb')
        read_path = json_path_test
        num = num_of_test
    with open(read_path, 'r') as load_f:
        for line in load_f:
            cnt = cnt + 1
            # json related
            load_dict = json.loads(line)
            _id = load_dict['id']
            _title = load_dict['title']
            _content = load_dict['content']
            soup = BeautifulSoup(_content, 'lxml')
            content = soup.get_text()
            title = _title
            # jieba_cut_for_search & remove stop words
            seg_list_title = remove_words(list(jieba.cut_for_search(title)))
            seg_list_content = remove_words(
                list(jieba.cut_for_search(content)))
            # convert list into string
            seg_str_title = " ".join(seg_list_title)
            seg_str_content = " ".join(seg_list_content)
            # dump file
            pickle.dump(seg_str_title, offile_for_title)
            pickle.dump(seg_str_content, offile_for_content)
            if(int((cnt * 100)/num) > int((cnt*100-100)/num)):
                print("dumping %d%%ok!\n" % (cnt * 100 / num))
    offile_for_title.close()
    offile_for_content.close()
    print("jieba_cut & dumping finished!\n")


def vectorize():
    # load sample
    print('load data...\n')
    te_cont_sample = load_data(write_path_test_content, p_test, p_x)
    tr_cont_sample = load_data(write_path_train_content, p_train, p_x)
    te_tit_sample = load_data(write_path_test_title, p_test, p_x)
    tr_tit_sample = load_data(write_path_train_title, p_train, p_x)
    print('loading x succeed!\n')
    y_train = load_data(train_y_path, p_train, p_y)
    print('loading y succeed!\n')
    # vectorizing
    print('vectorizing data start.\n')
    max_feature = 1000000
    count_vec = TfidfVectorizer(
        binary=False, decode_error='ignore', max_features=max_feature)
    x_train_content = count_vec.fit_transform(tr_cont_sample)
    x_test_content = count_vec.transform(te_cont_sample)
    x_train_title = 0.8 * count_vec.fit_transform(tr_tit_sample)
    x_test_title = 0.8 * count_vec.transform(te_tit_sample)
    x_train = hstack((x_train_content, x_train_title))
    x_test = hstack((x_test_content, x_test_title))
    print('vectorizing finished!\n')
    # dump data
    print('dumping train & test data.\n')
    ofile_test_x = open(save_path_test_x, 'wb')
    ofile_train_x = open(save_path_train_x, 'wb')
    ofile_train_y = open(save_path_train_y, 'wb')
    pickle.dump(x_test, ofile_test_x)
    ofile_test_x.close()
    pickle.dump(x_train, ofile_train_x)
    ofile_train_x.close()
    pickle.dump(y_train, ofile_train_y)
    ofile_train_y.close()
    print('dumping finished!\n')


def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l1')
    model.fit(train_x, train_y)
    return model


def training():
    # load file
    print('loading file...\n')
    x_train = load_file(save_path_train_x)
    y_train = load_file(save_path_train_y)
    x_test = load_file(save_path_test_x)
    print('loading finished!\n')
    # train & predict data
    print('training & predicting data...\n') 
    #model = logistic_regression_classifier(x_train, y_train)
    # 简单初始化xgb的分类器就可以
    clf =XGBClassifier(learning_rate=0.1, max_depth=10, n_estimators = 500, scale_pos_weight=4, silent=True, objective='binary:logistic')
    #param_test = {'n_estimators': list(range(30, 50, 2)),'max_depth': list(range(2, 7, 1))}
    #grid_search = GridSearchCV(verbose = 10, estimator = clf, param_grid = param_test, scoring='accuracy', cv=5)
    clf.fit(x_train, y_train, verbose = 10,eval_set=[(x_train, y_train)],eval_metric = 'auc')
    predict = clf.predict_proba(x_test)
    predict = predict[:, 1]
    print('training & predicting finished!\n')
    # scoring
    #score = model.score(x_train, y_train)
    #print("training score = %f" % score)
    # submitting
    print('submitting...\n')
    test_sampley = pd.read_csv(test_y_path)
    idno = test_sampley['id']
    My_prediction = pd.DataFrame({'id': idno, 'pred': predict})
    My_prediction.to_csv(submitting_path, index=False)


#jieba_cut(p_test)
#jieba_cut(p_train)
#vectorize()
training()
