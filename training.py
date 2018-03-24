from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import numpy as np
import pickle
import pandas as pd
import nltk
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

num_of_lines = 321910
test_num_of_lines = 40239
train_x_path = "E:\\Machine learning\\Task1\\source\\train_x.pkl"
train_y_path = "E:\\Machine learning\\Task1\\source\\train_y.pkl"
test_x_path = "E:\\Machine learning\\Task1\\source\\test_x.pkl"
test_y_path = "E:\\Machine learning\\Task1\\source\\test_y.pkl"

def load_data(path):
    ifile = open(path, 'rb')
    data = pickle.load(ifile)
    ifile.close()
    return data

# Multinomial Naive Bayes Classifier
def naive_bayes_classifier(train_x, train_y):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=1e-2)
    model.fit(train_x, train_y)
    return model

# SVM Classifier
def svm_classifier(train_x, train_y):
    from sklearn import svm
    model = svm.SVC()#(C=1.23, gamma=1.125, probability=True)
    model.fit(train_x, train_y)
    return model

# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l1')
    model.fit(train_x, train_y)
    return model

def xgb_classifier(train_x, train_y, test_x, test_y):
    from xgboost import XGBClassifier
    model = XGBClassifier(max_depth=100, 
                        learning_rate=0.01, 
                        n_estimators=100, 
                        silent=True, 
                        objective='binary:logistic', 
                        n_jobs = 4, 
                        gamma=0,
                        min_child_weight=1, 
                        max_delta_step=0, 
                        subsample=1, 
                        colsample_bytree=0.7, 
                        colsample_bylevel=1, 
                        alpha = 10,
                        scale_pos_weight=1, 
                        seed=1440, 
                        missing=None)
    model.fit(train_x, train_y,eval_metric = 'auc', verbose = True, eval_set = [(test_x, test_y)])
    return model

# load data
print('loading data...\n')
x_train = load_data(train_x_path)
y_train = load_data(train_y_path)
x_test = load_data(test_x_path)
print('loading finished.\n')

# train & predict data
print('training & predicting data...\n')
#x_train, x1_test, y_train, y1_test = train_test_split(x_train, y_train, test_size=0.01, random_state=1729)
model = logistic_regression_classifier(x_train, y_train)
#model = svm_classifier(x_train, y_train)
#model = naive_bayes_classifier(x_train, y_train)
#model = xgb_classifier(x_train, y_train, x1_test, y1_test)

#predict = model.predict(x_test)
predict = model.predict_proba(x_test)
predict = predict[:, 1]
print('training & predicting finished.\n')

# scoring
score = model.score(x_train, y_train)
print("training score = %f" % score)


# submitting
print('submitting...\n')
test_y_path = "E:\\Machine learning\\Task1\\test_data\\sample_submission.csv"
test_sampley = pd.read_csv(test_y_path)
idno = test_sampley['id']
My_prediction = pd.DataFrame({'id': idno, 'pred': predict})
My_prediction.to_csv(
    'E:\\Machine learning\\Task1\\submission\\submission1_logistic_regression.csv', index = False)
