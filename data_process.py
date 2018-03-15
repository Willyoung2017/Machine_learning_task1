import json
import re
from bs4 import BeautifulSoup
import jieba
import pickle

cnt = 0;
off_all = open("E:\\Machine learning\\Task1\\source\\cut_comments.pkl", 'wb')
with open("E:\\Machine learning\\Task1\\train_data\\train.json",'r') as load_f:
    for line in load_f:
        cnt = cnt + 1;
        load_dict = json.loads(line)
        #print(load_dict)
        train_id = load_dict['id']
        train_title = load_dict['title']
        train_content = load_dict['content']
        #print(train_content)
        soup = BeautifulSoup(train_content,'lxml')
        para = soup.get_text()
        seg_list = list(jieba.cut(para));
        pickle.dump(seg_list, off_all);
        if(cnt % 1000 == 0):
            print("%dok!\n",cnt)
        #print(train_title)
        #print(para)
off_all.close()

