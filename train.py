import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pyro
from pyro.distributions import Normal

import pandas as pd
import numpy as np

import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# 定义贝叶斯神经网络类
class BayesianNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BayesianNN, self).__init__()
        # 定义网络结构，这里使用一个简单的全连接层
        self.fc = nn.Linear(input_dim, output_dim)
        # 定义网络参数的先验分布，这里使用高斯分布
        self.prior_mean = nn.Parameter(torch.zeros_like(self.fc.weight), requires_grad=False)
        self.prior_std = nn.Parameter(torch.ones_like(self.fc.weight), requires_grad=False)
        # 定义网络参数的后验分布，这里使用高斯分布，其均值和方差是可学习的
        self.posterior_mean = nn.Parameter(torch.randn_like(self.fc.weight))
        self.posterior_std = nn.Parameter(torch.randn_like(self.fc.weight).exp())

    def forward(self, x):
        # 根据后验分布采样网络参数
        weight = self.posterior_mean + self.posterior_std * torch.randn_like(self.posterior_std)
        # 使用采样的参数进行前向传播
        output = torch.sigmoid(self.fc(x))
        # 计算后验分布和先验分布的对数似然比，作为KL散度的一部分
        log_posterior = torch.distributions.Normal(self.posterior_mean, self.posterior_std).log_prob(weight).sum()
        log_prior = torch.distributions.Normal(self.prior_mean, self.prior_std).log_prob(weight).sum()
        return output, log_posterior - log_prior

# 定义损失函数，这里使用均方误差加上KL散度的权衡
def loss_fn(output, target, kl, beta):
    mse = F.mse_loss(output, target)
    return mse + beta * kl



if __name__ == '__main__':
    datafile = 'data_building_4in.csv'
    data_s = pd.read_csv(datafile,sep=',')
    data_s = torch.tensor(np.array(data_s),dtype =torch.float32 )#[0:300]
    # data = torch.zeros(100,data_s.shape[1])

    # data_s = data_s[0:1000]
    # for i in range(100):
    #     data[i] = data_s[i]
    data_tt = data_s#[100:400]#
    # for i in range(data.shape[-1]):
    #     data[:,i] = (data[:,i]-data[:,i].min())/(data[:,i].max()-data[:,i].min())
    print(data_s[:,-1].max())
    print(data_s[:,-1].min())
    print(data_tt[:,-1].max()-data_tt[:,-1].min())
    i=-1
    lamda=1
    data_tt[:,i] = (data_tt[:,i]-data_tt[:,i].min())/(data_tt[:,i].max()-data_tt[:,i].min())*lamda
    # rate = 5
    # data = torch.zeros(3557//rate,data_s.shape[1])

    # for i in range(3557//rate):
    #     data[i] = data_tt[i*rate]
    data = data_tt[:int(data_tt.shape[0]*0.8)+1]
    print(data.shape)
    N = data.shape[-1] # size of data
    # data = torch.randn(N, /5) # example data
    x = data[:, :4]
    y = data[:, 4]

    # 定义超参数
    input_dim = 4 # 输入维度
    output_dim = 1 # 输出维度
    batch_size = 32 # 批次大小
    epochs = 100 # 训练轮数
    lr = 0.01 # 学习率
    beta = 0.1 # KL散度的权重

    # 加载原始数据，假设是一个N*5的数组data，前四列是输入，最后一列是输出
    x_data = data[:, :4]
    y_data = data[:, -1]

    # 将数据转换为张量，并划分为训练集和测试集
    x_tensor = x_data
    y_tensor = y_data
    train_size = int(0.8 * len(x_tensor)) # 训练集大小，这里使用80%的数据作为训练集
    test_size = len(x_tensor) - train_size # 测试集大小，剩下的数据作为测试集
    train_x, test_x = torch.split(x_tensor, [train_size, test_size])
    train_y, test_y = torch.split(y_tensor, [train_size, test_size])

    # 使用dataloader来打包batch训练
    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 创建贝叶斯神经网络实例，并定义优化器
    model = BayesianNN(input_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 开始训练
    for epoch in range(epochs):
        train_loss = 0.0 # 记录训练损失
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad() # 清空梯度
            # print(batch_x.shape)
            output, kl = model(batch_x) # 前向传播，得到输出和KL散度
            loss = loss_fn(output, batch_y, kl, beta) #
            print(loss)
            loss.backward()
            optimizer.step()
            torch.save(model.state_dict(), 'new/'+'model.pth')
