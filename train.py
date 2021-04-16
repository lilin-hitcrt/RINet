import torch
from net import RINet
from database import evalDataset,SigmoidDataset1
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import  metrics
def train():
    device=torch.device('cuda')
    net=RINet()
    # net.load('/home/l/workspace/python/test/model/08/model_test0.9675291730086251.pth')
    net.to(device=device)
    devicecpu=torch.device('cpu')
    print(net)
    train_dataset=SigmoidDataset1(['00','01','02','03','04','08','06','07','09','10'],1)
    test_dataset=evalDataset('05')
    batch_size=1024
    train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=8)
    test_loader=DataLoader(dataset=test_dataset,batch_size=4096,shuffle=False,num_workers=8)
    optimizer=torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr=0.01,weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[250,500,750],gamma=0.1)
    epoch=1000
    maxaccur=0.
    for i in range(epoch):
        net.train()
        pred=[]
        gt=[]
        epoch_loss=0
        batch_loss=0
        for i_batch,sample_batch in tqdm(enumerate(train_loader),total=len(train_loader),desc='Train epoch '+str(i)):
            optimizer.zero_grad()
            out=net(sample_batch["desc1"].to(device=device),sample_batch["desc2"].to(device=device))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(out, sample_batch["label"].to(device=device), reduction='sum')
            loss.backward()
            optimizer.step()
            # print("Loss:",loss)
            with torch.no_grad():
                batch_loss=loss.to(device=devicecpu).item()
                epoch_loss+=batch_loss
                outlabel = out.to(device=devicecpu).tolist()
                # print(outlabel)
                label = sample_batch['label'].to(device=devicecpu)
                label[label<0.9]=0
                label[label>0.1]=1
                # print(label)
                pred.extend(outlabel)
                gt.extend(label.tolist())
        scheduler.step()
        pred=np.array(pred,dtype='float32')
        gt=np.array(gt,dtype='float32')
        epoch_loss/=len(pred)
        print("Epoch ",i," loss:",epoch_loss)
        precision, recall, pr_thresholds = metrics.precision_recall_curve(gt,pred)
        F1_score = 2 * precision * recall / (precision + recall)
        F1_score = np.nan_to_num(F1_score)
        trainaccur = np.max(F1_score)
        print('train accur:',trainaccur)
        lastaccur=test(net=net,dataloader=test_loader,maxaccur=maxaccur,datatype="test")
        if lastaccur>maxaccur:
            maxaccur=lastaccur


def test(net,dataloader,datatype='test',maxaccur=0,save=True):
    net.eval()
    devicegpu = torch.device('cuda')
    devicecpu = torch.device('cpu')
    pred=[]
    gt=[]
    with torch.no_grad():
        for i_batch,sample_batch in tqdm(enumerate(dataloader),total=len(dataloader),desc="Eval"):
            out=net(sample_batch["desc1"].to(device=devicegpu),sample_batch["desc2"].to(device=devicegpu))
            out=out.to(device=devicecpu)
            outlabel=out
            label=sample_batch['label']
            pred.extend(outlabel)
            gt.extend(label)
        pred=np.array(pred,dtype='float32')
        gt=np.array(gt,dtype='float32')
        precision, recall, pr_thresholds = metrics.precision_recall_curve(gt,pred)
        F1_score = 2 * precision * recall / (precision + recall)
        F1_score = np.nan_to_num(F1_score)
        testaccur = np.max(F1_score)
        print(datatype,"accur:",testaccur)
        if save and testaccur>maxaccur:
            torch.save(net.state_dict(),'./model/model_'+datatype+str(testaccur)+'.pth')
        return testaccur

if __name__=='__main__':
    train()