import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def split_find(filename):
    file = open(filename,'r+t',encoding='utf-8')
    # file = open('sorted_demo_data','r+t',encoding='utf-8')
    offset = 0
    file.seek(0)
    x=[]
    y=[]
    index = 0
    for line in file.readlines():
        # print(offset)
        offset += len(line)
        k = line[:line.find(' ')]
        # if index % 10000 == 0:
            # k = k[:6]
        k = k[:16]
        ret = 0
        if int(k[:2]) == 9:
            ret += 1000000 * 60 * 60 * 24 * 31
        ret += int(k[2:4]) * 1000000 * 60 * 60 * 24
        ret += int(k[4:6]) * 1000000 * 60 * 60
        ret += int(k[6:8]) * 1000000 * 60
        ret += int(k[8:10]) * 1000000
        ret += int(k[10:16])
        x.append(ret)
        y.append(offset)
        index = index + 1
    #print(len(x))
    # mxx = max(x)
    # mxy = max(y)
    # mix = min(x)
    # miy = min(y)
    #for i in range(len(x)):
        # print(x[i])
    #    x[i] = (x[i] - mix) / (mxx - mix)
    #    y[i] = (y[i] - miy) / (mxy - miy)
    split=[]
    # print(x[:10])
    # print(y[:10])
    for index in range(2,len(x)):
        # xie = (y[index]-y[index-1])/(x[index]-x[index-1])
        xie = abs((x[index] - x[index - 1]) - (x[index - 1] - x[index - 2]))
        # print(xie)
        if xie > 50099532882003691251445549114761304427840143360:
            split.append(y[index])
    return split
    # print(len(split))
    # plt.plot(x,y, ls="-", lw=2, label="plot figure")
    # plt.legend()
    # plt.show()

split = split_find('sorted_dataset_2')
print(split)