import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import numpy as  np
import random

from torch.nn.modules.pooling import AdaptiveAvgPool1d, AvgPool1d,MaxPool1d
from database import SigmoidDataset
from torch.utils.data import  DataLoader

class RIConv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size):
        super(RIConv,self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.conv=nn.Sequential(nn.Conv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=1),nn.BatchNorm1d(out_channels),nn.LeakyReLU(negative_slope=0.1))
    
    def forward(self,x):
        x=F.pad(x,[0,self.kernel_size-1],mode='circular')
        out=self.conv(x)
        return out

class RIDowsampling(nn.Module):
    def __init__(self,ratio=2):
        super(RIDowsampling,self).__init__()
        self.ratio=ratio
    
    def forward(self,x):
        y=x[:,:,list(range(0,x.shape[2],self.ratio))].unsqueeze(1)
        for i in range(1,self.ratio):
            index=list(range(i,x.shape[2],self.ratio))
            y=torch.cat([y,x[:,:,index].unsqueeze(1)],1)
        norm=torch.norm(torch.norm(y,1,2),1,2)
        idx=torch.argmax(norm,1)
        idx=idx.unsqueeze(1).expand(x.shape[0],self.ratio)
        id_matrix=torch.tensor([list(range(self.ratio))]).expand(x.shape[0],self.ratio).to(device=x.device)
        out=y[id_matrix==idx]
        return out

class RINet(nn.Module):
    def __init__(self):
        super(RINet,self).__init__()
        self.conv1=nn.Sequential(RIConv(in_channels=12,out_channels=12,kernel_size=3),RIConv(in_channels=12,out_channels=16,kernel_size=3))
        self.conv2=nn.Sequential(RIDowsampling(3),RIConv(in_channels=16,out_channels=16,kernel_size=3))
        self.conv3=nn.Sequential(RIDowsampling(3),RIConv(in_channels=16,out_channels=32,kernel_size=3))
        self.conv4=nn.Sequential(RIDowsampling(2),RIConv(in_channels=32,out_channels=32,kernel_size=3))
        self.conv5=nn.Sequential(RIDowsampling(2),RIConv(in_channels=32,out_channels=64,kernel_size=3))
        self.conv6=nn.Sequential(RIDowsampling(2),RIConv(in_channels=64,out_channels=128,kernel_size=3))
        self.pool=AdaptiveAvgPool1d(1)
        self.linear=nn.Sequential(nn.Linear(in_features=288,out_features=128),nn.LeakyReLU(negative_slope=0.1),nn.Linear(in_features=128,out_features=1))
        
    def forward(self,x,y):
        fx=[]
        fy=[]
        x1=self.conv1(x)
        y1=self.conv1(y)
        fx.append(self.pool(x1).reshape(x.shape[0],-1))
        fy.append(self.pool(y1).reshape(x.shape[0],-1))
        x2=self.conv2(x1)
        y2=self.conv2(y1)
        fx.append(self.pool(x2).reshape(x.shape[0],-1))
        fy.append(self.pool(y2).reshape(x.shape[0],-1))
        x3=self.conv3(x2)
        y3=self.conv3(y2)
        fx.append(self.pool(x3).reshape(x.shape[0],-1))
        fy.append(self.pool(y3).reshape(x.shape[0],-1))
        x4=self.conv4(x3)
        y4=self.conv4(y3)
        fx.append(self.pool(x4).reshape(x.shape[0],-1))
        fy.append(self.pool(y4).reshape(x.shape[0],-1))
        x5=self.conv5(x4)
        y5=self.conv5(y4)
        fx.append(self.pool(x5).reshape(x.shape[0],-1))
        fy.append(self.pool(y5).reshape(x.shape[0],-1))
        x6=self.conv6(x5)
        y6=self.conv6(y5)
        fx.append(self.pool(x6).reshape(x.shape[0],-1))
        fy.append(self.pool(y6).reshape(x.shape[0],-1))
        featurex=torch.cat(fx,1)
        featurey=torch.cat(fy,1)
        diff=torch.abs(featurex-featurey)
        out=self.linear(diff).reshape(-1)
        if not self.training:
            out=torch.sigmoid(out)
        return out

    def load(self,model_file):
        dict=torch.load(model_file)
        self.load_state_dict(dict)




if __name__=="__main__":
    database=SigmoidDataset(['01','02','03','04','05','06','07','08','09','10'])
    test_loader=DataLoader(dataset=database,batch_size=32,shuffle=True,num_workers=8)
    net=RINet()
    # net.load('/home/l/workspace/python/test/model/model_test0.8818544366899302.pth')
    net.eval()
    a=np.random.random(size=[32,12,360])
    b=np.random.random(size=[32,12,360])
    c=np.roll(b,random.randint(1,360),2)
    # b=np.random.randint(low=0,high=1000,size=[32,12,360])
    a=torch.from_numpy(np.array(a,dtype='float32'))
    b=torch.from_numpy(np.array(b,dtype='float32'))
    c=torch.from_numpy(np.array(c,dtype='float32'))
    # print(a,b)
    out1=net(b,c)
    out2=net(c,b)
    print(out1)
    print(out2)
    # print(out1-out2)
    exit(1)
    for i,data in enumerate(test_loader):
        net(data['desc1'])