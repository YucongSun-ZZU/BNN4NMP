import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from net import BNN
from torch.utils.data import SequentialSampler
from mpl_toolkits import mplot3d
import pandas as pd
import seaborn as sns
import os
import torchbnn as bnn
from torchhk import transform_model

datafile = 'data_building.csv'#存储数据文件的文件名
data_s = pd.read_csv(datafile,sep=',')#读取名为 datafile 的 csv 文件，并将其存储为 DataFrame 对象 data_s，文件的分隔符是逗号。
data_s = torch.tensor(np.array(data_s),dtype =torch.float32 )#[0:300]#首先将pandas DataFrame对象data_s转换为numpy数组，然后使用np.array()函数将其转换为numpy数组类型，最后使用torch.tensor()函数将其转换为PyTorch张量类型
data_tt = data_s
print(data_s[:,-1].max())
i=-1
data_tt[:,i] = (data_tt[:,i]-data_tt[:,i].min())/(data_tt[:,i].max()-data_tt[:,i].min())*50
data = data_tt[:2846]
data.shape
class BNN(nn.Module):
    def __init__(self,in_features = 2,hidden = 8, out_features = 1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features,hidden),
            nn.PReLU(),
            nn.Linear(hidden,out_features),
        )
    def forward(self,x):
        return self.fc(x)
net = BNN()
transform_model(net, nn.Linear, bnn.BayesLinear, 
            args={"prior_mu":torch.mean(data[:,-1]), "prior_sigma":torch.var(data[:,-1]), "in_features" : ".in_features",
                  "out_features" : ".out_features", "bias":".bias"
                 }, 
            attrs={"weight_mu" : ".weight"})
net.load_state_dict(torch.load('best\data_building\80\80model.pth'))

data_s = pd.read_csv(datafile,sep=',')
data_s = torch.tensor(np.array(data_s),dtype =torch.float32 )#[0:300]
# data = torch.zeros(300,3)
# for i in range(300):
#     data[i] = data_s[i*10]
data = (data_s)#[100:400]

samples = 5000
y_samp = np.zeros((samples,1))
i = 2288
x = data[i][0:-1]
y = data[i][-1]
print(x)
print(y)
for s in range(samples):
    y_tmp = net(x).detach().numpy()
    y_samp[s] = y_tmp.reshape(-1)
y_samp = (y_samp*(data[:,-1].max().item()-data[:,-1].min().item())/ 50 + data[:,-1].min().item()) 
print((y_samp).mean())

y_samp = (y_samp).reshape(-1)

plt.figure(1)
sns.displot(np.where(((y_samp<9000.) &( y_samp>7000.)), y_samp,y))#直方图
plt.show()
# plt.figure(2)
# sns.kdeplot(np.where(((y_samp<9000.) &( y_samp>7000.)), y_samp,y),shade=True)#np.where(((y_samp<20.) &( y_samp>17.)), y_samp,y)
# # sns.kdeplot(np.array(l),shade=True)
# plt.axvline(y_samp.mean(), label='mean',linestyle='-.', color='r')
# plt.axvline(y, label='real',linestyle='-.', color='g')
# plt.axvline(np.percentile(y_samp.reshape(-1),97.5), label='real', color='b')
# print(np.percentile(y_samp.reshape(-1),97.5)-y_samp.mean())
# print(np.percentile(y_samp.reshape(-1),2.5)-y_samp.mean())
# plt.axvline(np.percentile(y_samp.reshape(-1),2.5), label='real', color='b')

