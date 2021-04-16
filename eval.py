import torch
from net import RINet
from database import evalDataset
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from sklearn import  metrics
from matplotlib import pyplot as plt
import sys

def eval(seq='00'):
    devicegpu=torch.device('cuda')
    devicecpu=torch.device('cpu')
    net=RINet()
    net.load('./model/05/model_test0.9254678778592534.pth')
    net.to(device=devicegpu)
    net.eval()
    test_dataset=evalDataset(seq)
    testdataloader=DataLoader(dataset=test_dataset,batch_size=4096,shuffle=False,num_workers=8)
    pred=[]
    gt=[]
    with torch.no_grad():
        for i_batch,sample_batch in tqdm(enumerate(testdataloader),total=len(testdataloader),desc="Eval seq "+str(seq)):
            out=net(sample_batch["desc1"].to(device=devicegpu),sample_batch["desc2"].to(device=devicegpu))
            out=out.to(device=devicecpu)
            outlabel=out.tolist()
            label=sample_batch['label']
            pred.extend(outlabel)
            gt.extend(label.tolist())
    with open(seq+'.txt','w') as f:
        for g,p in zip(gt,pred):
            f.write(str(p)+" "+str(g)+'\n')

    pred=np.array(pred,dtype='float32')
    gt=np.array(gt,dtype='float32')
    precision, recall, pr_thresholds = metrics.precision_recall_curve(gt,pred)
    plt.plot(recall, precision, color='darkorange',lw=2, label='P-R curve')
    plt.axis([0,1,0,1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('DL Precision-Recall Curve')
    plt.legend(loc="lower right")
    F1_score = 2 * precision * recall / (precision + recall)
    F1_score = np.nan_to_num(F1_score)
    F1_max_score = np.max(F1_score)
    print(F1_max_score)
    plt.show()

if __name__=='__main__':
    seq='00'
    if len(sys.argv)>1:
        seq=sys.argv[1]
    eval(seq)
