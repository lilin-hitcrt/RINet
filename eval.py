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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def eval(seq='00',model_path="/home/l/workspace/python/RINet/model/attention/kitti/"):
    net=RINet_attention()
    net.load(os.path.join(model_path,seq+'.pth'))
    net.to(device=device)
    net.eval()
    test_dataset=evalDataset(seq)
    testdataloader=DataLoader(dataset=test_dataset,batch_size=16384,shuffle=False,num_workers=8)
    pred=[]
    gt=[]
    with torch.no_grad():
        for i_batch,sample_batch in tqdm(enumerate(testdataloader),total=len(testdataloader),desc="Eval seq "+str(seq)):
            out=net(sample_batch["desc1"].to(device=device),sample_batch["desc2"].to(device=device))
            outlabel=out.cpu().tolist()
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

def recall(seq='00',model_path="/home/l/workspace/python/RINet/model/attention/kitti/"):
    pose_file="/media/l/yp2/KITTI/odometry/dataset/poses/"+seq+".txt"
    poses=np.genfromtxt(pose_file)
    poses=poses[:,[3,11]]
    inner=2*np.matmul(poses,poses.T)
    xx=np.sum(poses**2,1,keepdims=True)
    dis=xx-inner+xx.T
    dis=np.sqrt(np.abs(dis))
    id_pos=np.argwhere(dis<=5)
    id_pos=id_pos[id_pos[:,0]-id_pos[:,1]>50]
    pos_dict={}
    for v in id_pos:
        if v[0] in pos_dict.keys():
            pos_dict[v[0]].append(v[1])
        else:
            pos_dict[v[0]]=[v[1]]
    desc_file=os.path.join('./data/desc_kitti',seq+'.npy')
    descs=np.load(desc_file)
    descs/=50.0
    net=RINet_attention()
    net.load(os.path.join(model_path,seq+'.pth'))
    net.to(device=device)
    net.eval()
    # print(net)
    out_save=[]
    recall=np.array([0.]*25)
    for v in tqdm(pos_dict.keys()):
        candidates=[]
        targets=[]
        for c in range(0,v-50):
            candidates.append(descs[c])
            targets.append(descs[v])
        candidates=np.array(candidates,dtype='float32')
        targets=np.array(targets,dtype='float32')
        candidates=torch.from_numpy(candidates)
        targets=torch.from_numpy(targets)
        with torch.no_grad():
            out=net(candidates.to(device=device),targets.to(device=device))
            out=out.cpu().numpy()
            ids=np.argsort(-out)
            o=[v]
            o+=ids[:25].tolist()
            out_save.append(o)
            for i in range(25):
                if ids[i] in pos_dict[v]:
                    recall[i:]+=1
                    break
    np.savetxt(seq+'.txt',out_save,fmt='%d')
    recall/=len(pos_dict.keys())
    print(recall)
    plt.plot(list(range(1,len(recall)+1)),recall,marker='o')
    plt.axis([1,25,0,1])
    plt.xlabel('N top retrievals')
    plt.ylabel('Recall (%)')
    plt.show()


if __name__=='__main__':
    seq='08'
    if len(sys.argv)>1:
        seq=sys.argv[1]
    # eval(seq,"/home/l/workspace/python/RINet/model/attention/kitti/")
    recall(seq,"/home/l/workspace/python/RINet/model/attention/kitti/")
