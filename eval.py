import torch
from net import RINet,RINet_attention
from database import evalDataset, evalDataset_kitti360
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
    net=RINet_attention()
    net.load('/home/l/workspace/python/RINet/model/attention/kitti/00/model_test0.992849846782431.pth')
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
    pred=np.nan_to_num(pred)
    save_db=np.array([pred,gt])
    save_db=save_db.T
    if not os.path.exists('result'):
        os.mkdir('result')
    np.savetxt(os.path.join('result',seq+'.txt'),save_db,"%.4f")
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
