import numpy as np
from sklearn import datasets, linear_model
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
    return (x,y,mxy,miy)

x,y,mxy,miy = num_find('sorted_dataset_2')
print(len(x))
print(len(y))
st = 0
predict_y = []
lenn = 10000
for index in range(823):
    data = np.array(x[st:st+lenn]).reshape(-1,1)
    label = np.array(y[st:st+lenn]).reshape(-1, 1)
    LR = linear_model.LinearRegression()
    LR.fit(data, label)
    predict_y.extend(LR.predict(data))
    st = st + lenn
data = np.array(x[st:]).reshape(-1,1)
label = np.array(y[st:]).reshape(-1, 1)
LR = linear_model.LinearRegression()
LR.fit(data, label)
predict_y.extend(LR.predict(data))
print(len(predict_y))
plt.figure()
plt.plot(x,y,"-")
plt.plot(x,predict_y,"--")
plt.show()
error = []
for i in range(len(x)):
    tmp = abs(predict_y[i]-y[i])
    error.append(tmp*(mxy - miy) + miy)
print("max error:", max(error))
print(max(error[10000:20000]))
#plt.plot(x,error,"-")
#plt.show()
#plt.figure()
#plt.plot(x,y,"-")
#plt.plot(x,predict_y,"--")
#plt.plot(x[:10000],y[:10000],"-")
#plt.plot(x[:10000],predict_y[:10000],"--")
#plt.savefig('./test.jpg')
#plt.show()

