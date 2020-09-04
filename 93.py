import numpy as np
import math
from tqdm import tqdm
import time
from sklearn import datasets, linear_model
import random
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
                b.append(0.0)
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
    # for j in range(len(x)):
    #     predict_y_test.append(LR.coef_ * x[j] + LR.intercept_)
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
randomInput=inputx
random.shuffle(randomInput)
randomG=[]
randomC=[]
for i in range(len(randomInput)):
    k = randomInput[i]
    randomG.append(group(k, per))
    randomC.append(cul_time(k))
predict_y = []
start_time = time.time()
w, b = train(inputx,inputy,per,G,C)
end_time = time.time()
print("--------")
print("train:", end_time-start_time, "s")
offset_index = G[0]

start_time = time.time()
for g,c in zip(randomG,randomC):
    index = g - offset_index
    predict = w[index]*c+b[index]
end_time = time.time()
print("query:", end_time - start_time, "s")

print("per:",per)
print("size:",len(w))
print("--------")

