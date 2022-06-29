import torch
from torch import nn
import numpy as np
import random
import math
import Net
import matplotlib.pyplot as plt


class Params():
    def __init__(self):
        self.data = np.loadtxt("exchange_rate.txt", delimiter=",")
        self.net = "LSTNet_Skip"  # LSTNet_Skip,LSTNet_Att,LSTNet_without_Skip
        # LSTNet_without_CNN,LSTNet_without_AR,LSTNet_AttSKip
        self.epoch = 50
        self.window_size = 28
        self.batch_size = 128
        self.kernel_size = 5
        self.horizon = 12  # 3,6,12,24
        self.hiddenC = 50
        self.hiddenG = 50
        self.hiddenS = 50
        self.T = 7  # 7,24
        self.ar_window = 4
        self.dropout = 0.2
        self.train = 0.6
        self.valid = 0.2
        self.lr = 0.001
        self.save = "exchange_rate_Skip.pt"
        self.device = "cuda:0"
        self.istrain=False

def Normalization(data):  # 对数据进行归一化预处理
    _, data_dim = data.shape
    res = np.ones(data.shape)
    scale = np.zeros(data_dim)
    for i in range(data_dim):  # 将数据归一化到(-1,1)之间
        scale[i] = np.max(np.abs(data[:, i]))
        res[:, i] = data[:, i] / scale[i]
    return res, scale


def Recover(data, scale):  # 将归一化数据还原为原数据
    res = np.zeros(data.shape)
    for i in range(data.shape[1]):
        res[:, i] = data[:, i] * scale[i]
    return res


def DivideData(data, train, valid):  # 给定数据,划分成训练集，测试集，验证集
    data_num, data_dim = data.shape
    train_num = int(data_num * train)
    valid_num = int(data_num * valid)
    data_train = data[:train_num, :]
    data_valid = data[train_num:train_num + valid_num, :]
    data_test = data[train_num + valid_num:, :]
    return data_train, data_valid, data_test


def Get_epoch_random(data, window_size, horizon, batch_size):  # 随机获取batch_size个样本作为epoch 用于训练模型
    data_train = data[random.randint(0, window_size):]  # 每次随机选择训练集数据开始点
    train_num, train_dim = data_train.shape  # train_num:训练集数据的数目 train_dim:训练集数据的维度
    window_num = (train_num - 1) // window_size  # 窗口的总数量
    initial_indices = list(range(window_size + horizon, window_num * window_size, window_size))  # 窗口的起点下标索引
    random.shuffle(initial_indices)  # 打乱下标索引
    batch_num = train_num // batch_size  # batch的数量
    for i in range(0, batch_size * batch_num, batch_size):
        initial_indices_per_batch = initial_indices[i: i + batch_size]  # batch_size个window索引
        X_list = [data_train[j - window_size - horizon:j - horizon, :] for j in
                  initial_indices_per_batch]  # 从索引下标取出数据，保存到列表中
        Y_list = [data_train[j, :] for j in initial_indices_per_batch]
        X = torch.zeros(batch_size, window_size, train_dim)  # 将列表转化为tensor
        Y = torch.zeros(batch_size, train_dim)
        X = X.cuda()
        Y = Y.cuda()
        for i in range(len(X_list)):
            X[i, :, :] = torch.from_numpy(X_list[i])
            Y[i, :] = torch.from_numpy(Y_list[i])
        yield torch.tensor(X), torch.tensor(Y)


def train(net, data_train, data_valid, data_test, window_size, horizon, batch_size, num_epochs, lr, device):
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.MSELoss()
    max_loss = 1000000
    for epoch in range(num_epochs):
        train_iters = Get_epoch_random(data_train, window_size, horizon, batch_size)    #随机顺序进行训练
        net.train()
        for X, Y in train_iters:
            optimizer.zero_grad()
            X, Y = X.to(device), Y.to(device)
            Y_hat = net(X)
            l = loss(Y_hat, Y)
            l.backward()
            optimizer.step()
        valid_loss = evaluate(net, data_train, data_valid, data_test, window_size, horizon, batch_size, epoch)
        if valid_loss < max_loss:  # 将验证集误差最小的模型保存
            with open(params.save, 'wb') as f:
                torch.save(net, f)
            max_loss = valid_loss


def evaluate(net, data_train, data_valid, data_test, window_size, horizon, batch_size, epoch):  #评估训练集 测试集 验证集的RSE以及CORR
    net.eval()
    loss = nn.MSELoss()
    train_iters = Get_epoch_sequence(data_train, window_size, horizon, batch_size)
    valid_iters = Get_epoch_sequence(data_valid, window_size, horizon, batch_size)
    test_iters = Get_epoch_sequence(data_test, window_size, horizon, batch_size)
    total_loss = 0
    num = 0
    pr = (epoch % 5 == 0)
    for X, Y in train_iters:
        Y_hat = net(X)
        l = loss(Y_hat, Y)
        total_loss += l
        num += 1
    if pr:
        print("epoch{0}:".format(epoch))
        print("train loss:{0}".format(total_loss / num))
    pred_y = None
    true_y = None
    for X, Y in valid_iters:
        Y_hat = net(X)
        if pred_y is None:
            pred_y = Y_hat
            true_y = Y
        else:
            pred_y = torch.cat((pred_y, Y_hat))
            true_y = torch.cat((true_y, Y))
        num = num + 1

    total_loss_valid = loss(pred_y, true_y)
    valid_rse = math.sqrt(total_loss_valid / num) / data_valid.std()
    if pr:
        print("valid rse:{0}".format(valid_rse))

    predict = pred_y.data.cpu().numpy()
    Ytest = true_y.data.cpu().numpy()
    sigma_p = predict.std(axis=0)
    sigma_g = Ytest.std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    valid_correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    valid_correlation = (valid_correlation[index]).mean()
    if pr:
        print("valid cor:{0}".format(valid_correlation))
    pred_y = None
    true_y = None
    for X, Y in test_iters:
        Y_hat = net(X)
        if pred_y is None:
            pred_y = Y_hat
            true_y = Y
        else:
            pred_y = torch.cat((pred_y, Y_hat))
            true_y = torch.cat((true_y, Y))
        num = num + 1

    total_loss = loss(pred_y, true_y)
    test_rse = math.sqrt(total_loss / num) / data_test.std()
    if pr:
        print("test rse:{0}".format(test_rse))
    predict = pred_y.data.cpu().numpy()
    Ytest = true_y.data.cpu().numpy()
    sigma_p = predict.std(axis=0)
    sigma_g = Ytest.std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    test_correlation = (correlation[:]).mean()
    if pr:
        print("test cor:{0}\n\n".format(test_correlation))
    return total_loss_valid


def Get_epoch_sequence(data, window_size, horizon, batch_size):  # 按样本自然顺序，获取epoch 用于评估模型
    data_num, data_dim = data.shape
    data_set = range(window_size + horizon - 1, data_num)
    length = len(data_set)
    X = torch.zeros((length, window_size, data_dim))
    Y = torch.zeros((length, data_dim))
    for i in range(length):
        end = data_set[i] - horizon + 1
        start = end - window_size
        X[i, :, :] = torch.from_numpy(data[start:end, :])
        Y[i, :] = torch.from_numpy(data[data_set[i], :])
    start = 0
    while start < length:
        end = min(length, start + batch_size)
        X_out = X[start:end, :, :]
        Y_out = Y[start:end, :]
        start = start + batch_size
        X_out = X_out.cuda()
        Y_out = Y_out.cuda()
        yield X_out, Y_out


def NET(params):
    net = params.net
    if net == "LSTNet_Skip":
        return Net.LSTNet_Skip(params)
    elif net == "LSTNet_Att":
        return Net.LSTNet_Att(params)
    elif net == "LSTNet_without_Skip":
        return Net.LSTNet_without_Skip(params)
    elif net == "LSTNet_without_CNN":
        return Net.LSTNet_without_CNN(params)
    elif net == "LSTNet_without_AR":
        return Net.LSTNet_without_AR(params)
    elif net == "LSTNet_AttSkip":
        return Net.LSTNet_AttSkip(params)
    else:
        return None


def Compare(scale):     #绘制有AR组件和无AR组件的对比图
    with open("exchange_rate_ar.pt", 'rb') as f:
        net_withoutAR = torch.load(f)
    with open("exchange_rate.pt", 'rb') as f:
        net = torch.load(f)
    pred_y = None
    pred_y_ar = None
    true_y = None
    iters = Get_epoch_sequence(params.data, params.window_size, params.horizon, params.batch_size)
    for X, Y in iters:
        Y_net = net(X)
        Y_net_ar = net_withoutAR(X)
        if pred_y is None:
            pred_y = Y_net
            pred_y_ar = Y_net_ar
            true_y = Y
        else:
            pred_y = torch.cat((pred_y, Y_net))
            true_y = torch.cat((true_y, Y))
            pred_y_ar = torch.cat((pred_y_ar, Y_net_ar))
    true_y = Recover(true_y.cpu().detach().numpy(), scale)
    pred_y = Recover(pred_y.cpu().detach().numpy(), scale)
    pred_y_ar = Recover(pred_y_ar.cpu().detach().numpy(), scale)
    p1, = plt.plot(true_y[:, 3], 'r')
    p2, = plt.plot(pred_y[:, 3], 'b')
    #p3, = plt.plot(pred_y_ar[:, 3], 'g')
    plt.legend([p2, p1], ["LSTnet_pred", "True_y"], loc='upper left')
    #plt.legend([p3, p2, p1], ["without_Ar_pred", "LSTnet_pred", "True_y"], loc='upper left')
    plt.savefig('compare.jpg')
    plt.show()


if __name__ == "__main__":
    params = Params()
    params.data, scale = Normalization(params.data)
    net = NET(params)
    data_train, data_valid, data_test = DivideData(params.data, params.train, params.valid)
    if params.istrain:
        train(net, data_train, data_valid, data_test, params.window_size, params.horizon, params.batch_size,
              params.epoch, params.lr, params.device)
    with open(params.save, 'rb') as f:
        net = torch.load(f)

    evaluate(net, data_train, data_valid, data_test, params.window_size, params.horizon, params.batch_size,
             params.epoch)
    Compare(scale)
