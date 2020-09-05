import numpy as np
import math
import time
from sklearn import datasets, linear_model
from BTrees.OIBTree import OIBTree
from dataFS import DataFS
import random
import matplotlib.pyplot as plt
from multi import MultiProcess

def cul_time(k):
    k = k[:16]
    ret = 0
    ret += int(k[8:10]) * 1000000
    ret += int(k[10:16])
    return ret

def group(k):
    k = k[:10]
    ret = 0
    if int(k[:2]) == 9:
        ret += 60 * 60 * 24 * 31
    ret += int(k[2:4]) * 60 * 60 * 24
    ret += int(k[4:6]) * 60 * 60
    ret += int(k[6:8]) * 60
    ret += int(k[8:10])
    return math.floor(ret/6)-294871

def Cul_GC(inputx, inputy):
    G = []
    C = []
    for i in range(len(inputx)):
        G.append(group(inputx[i]))
        C.append(cul_time(inputx[i]))
    return G,C

def train(inputx, inputy, G, C):
    x=[]
    y=[]
    w = []
    b = []
    pre_ret = G[0]
    pos = 0
    for i in range(len(inputx)):
        offset = inputy[i]
        k = inputx[i]
        ret = G[i]
        if ret != pre_ret:
            data = np.array(x).reshape(-1, 1)
            label = np.array(y).reshape(-1, 1)
            LR = linear_model.LinearRegression()
            LR.fit(data, label)
            w.append(LR.coef_)
            b.append(LR.intercept_)
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

def predict(k,btree,w,b):
    data = DataFS()
    if k in btree.keys():
        v = data.query(k, btree[k])
        return v
    index = group(k)
    offset = w[index]*cul_time(k)+b[index]
    v = data.query(k,offset)
    return v

def load_model():
    data = DataFS()
    keys, label = data.gen_train_data()
    btree = OIBTree()
    rubbish_k = np.load('6k.npy')
    rubbish_v = np.load('6v.npy')
    w = np.load('w.npy', allow_pickle=True)
    b = np.load('b.npy', allow_pickle=True)
    for i in range(len(rubbish_k)):
        btree[str(rubbish_k[i])] = int(rubbish_v[i])
    return keys, label,  w, b, btree

def process_run(mp, num, tot, keys, btree, w, b):
    times = int(tot/num)
    for i in range(num):
        mp.add(i, predict, keys[i * times:(i + 1) * times], btree, w, b)

# import sys
# print(sys.argv)
# if __name__=='__main__':
#     input,w,b,btree = load_model()
#     keys = input[:int(sys.argv[2])]
#     random.shuffle(keys)
#     time_start = time.time()
#     threads=[]
#     mp = MultiProcess()
#     process_run(mp,int(sys.argv[1]),int(sys.argv[2]),keys,btree,w,b)
#     results=mp.get_res()
#     res=[]
#     print(len(results))
#     for result in results:
#         res.extend(result)
#     time_end = time.time()
#     print(len(res))
#     print('predict:', time_end-time_start,'s')

#point search:
input,label,w,b,btree = load_model()
keys = input[:100000]
y = label[:100000]
for i in keys:
    print(predict(i,btree,w,b))


# with open("btree.pkl","wb") as file:
#     pickle.dump(btree,file,True)



