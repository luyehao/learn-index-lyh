import BasicFS
import NN
import torch

import matplotlib.pyplot as plt

if __name__ == "__main__":
    torch.random.manual_seed(19260817)

    bs = BasicFS.BasicFS('sorted_demo_data')
    x, y = bs.gen_test_data()
    mxx = max(x)
    mxy = max(y)
    mix = min(x)
    miy = min(y)
    # print(mxx)
    for i in range(len(x)):
        # print(x[i])

        x[i] = (x[i] - mix) / (mxx - mix)
        y[i] = (y[i] - miy) / (mxy - miy)

    x = torch.tensor(x, requires_grad=False).view(-1, 1)
    y = torch.tensor(y, requires_grad=False).view(-1, 1)

    net = NN.NeuralNet()

    for j in range(1):
        for i in range(1000):
            loss = NN.run_session(net, x, y, True)

        loss, predict = NN.run_session(net, x, y, False)

        # plt.plot(x.view(-1), predict.view(-1))
        # plt.plot(x.view(-1), y.view(-1))
        # plt.figure()
        plt.plot(x.view(-1), ((y-predict)*(mxy-miy)+miy).view(-1))
        plt.show()

        print(loss)