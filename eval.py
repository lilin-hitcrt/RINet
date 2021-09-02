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
import time
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device=torch.device("cpu")
def eval(seq='00',model_path="/home/l/workspace/python/RINet/model/attention/kitti/"):
    net=RINet_attention()
    net.load(os.path.join(model_path,seq+'.pth'))
    # net.load(os.path.join("/home/l/workspace/python/RINet/model/attention1/00.pth"))
    net.to(device=device)
    net.eval()
    test_dataset=evalDataset(seq)
    # test_dataset=evalDataset_kitti360(seq)
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

def fast_eval(seq='00',model_path="/home/l/workspace/python/RINet/model/attention/kitti/",disc_path='./data/desc_kitti',pair_path='./data/pairs_kitti/neg_100'):
    net=RINet_attention()
    net.load(os.path.join(model_path,seq+'.pth'))
    net.to(device=device)
    net.eval()
    print(net)
    desc_file=os.path.join(disc_path,seq+'.npy')
    desc_o=np.load(desc_file)/50.0
    descs_torch=torch.from_numpy(desc_o).to(device)
    total_time=0.
    with torch.no_grad():
        time1=time.time()
        descs=net.gen_feature(descs_torch).cpu().numpy()
        total_time+=(time.time()-time1)
    print("feature time:",total_time)
    pair_file=os.path.join(pair_path,seq+'.txt')
    pairs=np.genfromtxt(pair_file,dtype='int32').reshape(-1,3)
    # desc1=descs[pairs[:,0]]
    # desc2=descs[pairs[:,1]]
    # diff=desc1-desc2
    # diff=1./np.sum(diff*diff,axis=1)
    # diff=diff.reshape(-1,1)
    # diff=np.nan_to_num(diff)
    # label=pairs[:,2].reshape(-1,1)
    # precision, recall, pr_thresholds = metrics.precision_recall_curve(label, diff)
    desc1=torch.from_numpy(descs[pairs[:,0]]).to(device)
    desc2=torch.from_numpy(descs[pairs[:,1]]).to(device)
    total_time=0
    with torch.no_grad():
        time1=time.time()
        scores=net.gen_score(desc1,desc2).cpu().numpy()
        total_time+=(time.time()-time1)
    print("score time:",total_time)
    gt=pairs[:,2].reshape(-1,1)
    precision, recall, pr_thresholds = metrics.precision_recall_curve(gt, scores)
    F1_score = 2 * precision * recall / (precision + recall)
    F1_score = np.nan_to_num(F1_score)
    F1_max_score = np.max(F1_score)
    print(F1_max_score)
    plt.plot(recall, precision, color='darkorange',lw=2, label='P-R curve')
    plt.axis([0,1,0,1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
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
    if not os.path.exists('result'):
        os.mkdir('result')
    np.savetxt(os.path.join('result',seq+'_recall.txt'),out_save,fmt='%d')
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
    fast_eval(seq='00',model_path="/home/l/workspace/python/RINet/model/attention1/",disc_path='./data/desc_kitti',pair_path='./data/pairs_kitti/neg_100')
    # recall(seq,"/home/l/workspace/python/RINet/model/attention/kitti/")
