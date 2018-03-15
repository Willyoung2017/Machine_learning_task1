import pickle
import numpy as np

num_of_lines = 321910
test_num_of_lines = 40239

def read_data(path):
    file = open(path, 'r', encoding='UTF-8')
    line = file.readlines()
    for i in range(len(line)):
        line[i] = line[i].strip('\n')
    file.close()
    return line


ifile = open("E:\\Machine learning\\Task1\\source\\test_cut_texts.pkl", 'rb')
ofile = open("E:\\Machine learning\\Task1\\source\\test_processed_texts.pkl", 'wb')
stop_word_list = read_data(
    "E:\\Machine learning\\Task1\\source\\stop_word.txt")
stop_word_list.append('\n')
#processed_list = list(range(num_of_lines))
for i in range(test_num_of_lines):
    _line = pickle.load(ifile)
    line = list(set(_line))
    #line.sort(key=_line.index)
    for j in stop_word_list:
        if j in line:
            line.remove(j)
    Line = line.copy()
    for word in Line:
        if word[0] >= '0' and word[0] <= '9':
            line.remove(word)
    #processed_list[i] = line
    pickle.dump(line, ofile)
    if i % 100 == 0:
        print("%dfinished\n" % i)
        #print(line, '\n')
ifile.close()
ofile.close()
