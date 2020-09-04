import numpy as np
import math
from tqdm import tqdm
import time
from sklearn import datasets, linear_model
from BTrees import OIBTree
import matplotlib.pyplot as plt

class DataProvider():
    def __init__(self, filename):
        self.file = open(filename, 'r+t', encoding='utf-8')

    def gen_test_data(self) -> (list, list):
        offset = 0
        self.file.seek(0)
        X = []
        Y = []
        for line in tqdm(self.file.readlines(), desc='Load data'):
            k = line[:line.find(' ')]
            X.append(k)
            Y.append(offset)
            offset += len(line)
        return X, Y

def cul_time(k):
    k = k[:16]
    ret = 0
    ret += int(k[8:10]) * 1000000
    ret += int(k[10:16])
    return ret

def group(k,per):
    k = k[:10]
    ret = 0
    if int(k[:2]) == 9:
        ret += 60 * 60 * 24 * 31
    ret += int(k[2:4]) * 60 * 60 * 24
    ret += int(k[4:6]) * 60 * 60
    ret += int(k[6:8]) * 60
    ret += int(k[8:10])
    return math.floor(ret/per)

def train(inputx, inputy, per, G, C):
    predict_y = []
    x=[]
    y=[]
    w = []
    b = []
    pre_ret = G[0]
    pos = 0
    predict_y_test = []
    for i in range(len(inputx)):
        offset = inputy[i]
        k = inputx[i]
        ret = G[i]
        if ret != pre_ret:
            data = np.array(x).reshape(-1, 1)
            label = np.array(y).reshape(-1, 1)
            LR = linear_model.LinearRegression()
            LR.fit(data, label)
            # predict_y.extend(LR.predict(data))
            w.append(LR.coef_)
            b.append(LR.intercept_)
            # for j in range(len(x)):
            #     predict_y_test.append(LR.coef_*x[j]+LR.intercept_)
            # print(len(data))
            for j in range(ret - pre_ret - 1):
                w.append(0.0)
                b.append(0)
            x = []
            y = []
            x.append(C[i])
            y.append(offset)
            pre_ret = ret
            pos = pos + 1
        else:
            x.append(C[i])
            y.append(offset)

    data = np.array(x).reshape(-1, 1)
    label = np.array(y).reshape(-1, 1)
    LR = linear_model.LinearRegression()
    LR.fit(data, label)
    w.append(LR.coef_)
    b.append(LR.intercept_)
    return w, b


data=DataProvider('unique_time_stamp')
inputx, inputy = data.gen_test_data()
per = 6
G = []
C = []
for i in range(len(inputx)):
    offset = inputy[i]
    k = inputx[i]
    G.append(group(k, per))
    C.append(cul_time(k))
predict_y = []
start_time = time.time()
w, b = train(inputx,inputy,per,G,C)
end_time = time.time()

np.save('w',w)
np.save('b',b)

print("--------")
print("train:", end_time-start_time, "s")
start_time = time.time()
offset_index = G[0]
print("start:", offset_index)
for i in range(len(inputx)):
    index = G[i] - offset_index
    predict_y.append(w[index]*C[i]+b[index])
end_time = time.time()
print("query:", end_time - start_time, "s")
error = []
num_error = 0
filename1 = str(per) + 'k'
filename2 = str(per) + 'v'
save_x = []
save_y = []
for i in range(len(inputx)):
    error.append(abs(predict_y[i]-inputy[i]))
    if abs(predict_y[i]-inputy[i]) > 4096:
        num_error = num_error + 1
        save_x.append(inputx[i])
        save_y.append(inputy[i])
np.save(filename1,save_x)
np.save(filename2,save_y)
print("per:",per)
print("size:",len(w))
print("max error:", max(error))
print("error number:", num_error)
print("accept:", len(inputx)-num_error)
print((len(inputx)-num_error)/len(inputx))
print("--------")

