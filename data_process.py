import json
import re
from bs4 import BeautifulSoup

with open("E:\\Machine learning\\Task1\\train_data\\train.json",'r') as load_f:
    for line in load_f:
        load_dict = json.loads(line)
        #print(load_dict)
        train_id = load_dict['id']
        train_title = load_dict['title']
        train_content = load_dict['content']
        print(train_content)
        soup = BeautifulSoup(train_content,'lxml')
        para = soup.get_text()
        print(train_title)
        print(para)
        #print(test_content)
        #para = re.findall(r"<p>(.*?)</p>",test_content)
        #para = para.replace('<strong>','')
        #print(para)
'''
        soup = BeautifulSoup(test_content,'lxml')
        print(type(soup.findAll('p')))
        for para in soup.findAll('p'):
            print(para.get_text())    
        print(test_id)
      
'''
