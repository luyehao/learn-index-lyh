import numpy as np
import math
from tqdm import tqdm
from sklearn import datasets, linear_model
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

def train(inputx, inputy, per):
    predict_y = []
    x=[]
    y=[]
    w = []
    b = []
    pre_ret = group(inputx[0], per)
    pos = 0
    predict_y_test = []
    for i in range(len(inputx)):
        offset = inputy[i]
        k = inputx[i]
        ret = group(k,per)
        if ret != pre_ret:
            data = np.array(x).reshape(-1, 1)
            label = np.array(y).reshape(-1, 1)
            LR = linear_model.LinearRegression()
            LR.fit(data, label)
            first_x = x[0]
            first_y = y[0]
            last_x = x[-1]
            last_y = y[-1]
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
            x.append(cul_time(k))
            y.append(offset)
            pre_ret = ret
            pos = pos + 1
        else:
            x.append(cul_time(k))
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
for per in range(11):
    if per == 0:
        continue
    predict_y = []
    w, b = train(inputx,inputy,per)
    offset_index = group(inputx[0], per)
    for i in range(len(inputx)):
        index = group(inputx[i],per) - offset_index
        predict_y.append(w[index]*cul_time(inputx[i])+b[index])
    error = []
    num_error = 0
    for i in range(len(inputx)):
        error.append(abs(predict_y[i]-inputy[i]))
        if abs(predict_y[i]-inputy[i]) > 4096:
            num_error = num_error + 1
    print("--------")
    print("per:",per)
    print("size:",len(w))
    print("max error:", max(error))
    print("error number:", num_error)
    print("accept:", len(inputx)-num_error)
    print((len(inputx)-num_error)/len(inputx))
    print("--------")

