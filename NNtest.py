import torch
from torch.autograd import Variable
import os
import logging
import pandas as pd
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def num_find(filename):
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
    mxx = max(x)
    mxy = max(y)
    mix = min(x)
    miy = min(y)
    for i in range(len(x)):
        x[i] = (x[i] - mix) / (mxx - mix)
        y[i] = (y[i] - miy) / (mxy - miy)
    #plt.plot(x, y, ls="-", lw=2, label="plot figure")
    #plt.legend()
    #plt.show()
    return (x, y, mxy, miy)

def predict(x,y):
    n = len(x)
    mean_x = np.mean(x)
    sum_val = np.sum(x)*np.sum(x)/n
    up = 0.0
    down = 0.0
    for i in range(n):
        up = up + y[i]*(x[i]-mean_x)
        down = down + x[i]*x[i]
    w = up/(down-sum_val)
    b=0.0
    for i in range(n):
        b = b + y[i] - w*x[i]
    b = b/n
    predict_x = x
    predict_y = []
    for i in range(n):
        predict_y.append(w * x[i] + b)
    #plt.plot(predict_x, predict_x, ls="-", lw=2, label="plot figure")
    #plt.legend()
    #plt.show()
    return (predict_x,predict_y)

def train(x, y):
    print('------      构建数据集      ------')
    # Variable是将tensor封装了下，用于自动求导使用
    x, y = Variable(x), Variable(y)
    # 绘图展示
    plt.scatter(x.data.numpy(), y.data.numpy())
    # plt.show()

    print('------      搭建网络      ------')

    # 使用固定的方式继承并重写 init和forword两个类
    class Net(torch.nn.Module):
        def __init__(self, n_feature, n_hidden, n_output):
            # 初始网络的内部结构
            super(Net, self).__init__()
            self.hidden = torch.nn.Linear(n_feature, n_hidden)
            self.predict = torch.nn.Linear(n_hidden, n_output)

        def forward(self, x):
            # 一次正向行走过程
            x = F.relu(self.hidden(x))
            x = self.predict(x)
            return x

    net = Net(n_feature=1, n_hidden=1000, n_output=1)
    print('网络结构为：', net)

    print('------      启动训练      ------')
    loss_func = F.mse_loss
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

    # 使用数据 进行正向训练，并对Variable变量进行反向梯度传播  启动100次训练
    for t in range(10000):
        # 使用全量数据 进行正向行走
        prediction = net(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()  # 清除上一梯度
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 应用梯度

        # 间隔一段，对训练过程进行可视化展示
        if t % 1000 == 0:
            plt.cla()
            plt.scatter(x.data.numpy(), y.data.numpy())  # 绘制真是曲线
            plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
            plt.text(0.5, 0, 'Loss=' + str(loss.item()), fontdict={'size': 20, 'color': 'red'})
            plt.pause(0.1)
    plt.ioff()
    plt.show()
    print('------      预测和可视化      ------')
    return y.data.numpy()

x,y,mxy,miy = num_find('sorted_dataset_2')
predict_x,predict_y = predict(x,y)
error = []
for i in range(len(x)):
    tmp = predict_y[i]-y[i]
    error.append(tmp*(mxy - miy) + miy)
inputx = torch.FloatTensor([x]).t()
inputy = torch.FloatTensor([error]).t()
#print(inputx.size())
#print(inputy.size())
error_predict = train(inputx, inputy)
#error = error - error_predict
#plt.plot(x,error,"-")
#plt.show()

