import torch
from net import RINet,RINet_attention
from database import evalDataset,SigmoidDataset,evalDataset_kitti360,SigmoidDataset_kitti360,SigmoidDataset_train,SigmoidDataset_eval
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import  metrics
from tensorboardX import SummaryWriter
def train():
    writer = SummaryWriter()
    device=torch.device('cuda')
    net=RINet_attention()
    net.to(device=device)
    print(net)
    # train_dataset=SigmoidDataset_train(['00','01','02','03','04','05','06','07','08','09','10'],1)
    # test_dataset=SigmoidDataset_eval(['00','01','02','03','04','05','06','07','08','09','10'],1)
    train_dataset=SigmoidDataset(['00','01','03','04','05','06','07','08','09','10'],1)
    test_dataset=evalDataset('02')
    # train_dataset=SigmoidDataset_kitti360(['0009','0003','0007','0002','0004','0006','0000','0010'],1)
    # test_dataset=evalDataset_kitti360('0005')
    batch_size=1024
    train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=6)
    test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,num_workers=6)
    optimizer=torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr=0.02,weight_decay=1e-6)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,1,0.5)
    epoch=1000
    maxaccur=0.
    batch_num=0
    for i in range(epoch):
        net.train()
        pred=[]
        gt=[]
        for i_batch,sample_batch in tqdm(enumerate(train_loader),total=len(train_loader),desc='Train epoch '+str(i),leave=False):
            optimizer.zero_grad()
            out=net(sample_batch["desc1"].to(device=device),sample_batch["desc2"].to(device=device))
            labels=sample_batch["label"].to(device=device)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(out, labels)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                writer.add_scalar('total loss',loss.cpu().item(),global_step=batch_num)
                batch_num+=1
                outlabel = out.cpu().numpy()
                label = sample_batch['label'].cpu().numpy()
                mask=(label>0.9906840407)|(label<0.0012710163)
                label=label[mask]
                label[label<0.5]=0
                label[label>0.5]=1
                pred.extend(outlabel[mask].tolist())
                gt.extend(label.tolist())
        scheduler.step()
        pred=np.array(pred,dtype='float32')
        pred=np.nan_to_num(pred)
        gt=np.array(gt,dtype='float32')
        precision, recall, pr_thresholds = metrics.precision_recall_curve(gt,pred)
        F1_score = 2 * precision * recall / (precision + recall)
        F1_score = np.nan_to_num(F1_score)
        trainaccur = np.max(F1_score)
        print('Train F1:',trainaccur)
        writer.add_scalar('train f1',trainaccur,global_step=i)
        lastaccur=test(net=net,dataloader=test_loader,maxaccur=maxaccur,datatype="test")
        writer.add_scalar('eval f1',lastaccur,global_step=i)
        if lastaccur>maxaccur:
            maxaccur=lastaccur


def test(net,dataloader,datatype='test',maxaccur=0,save=True):
    net.eval()
    devicegpu = torch.device('cuda')
    pred=[]
    gt=[]
    with torch.no_grad():
        for i_batch,sample_batch in tqdm(enumerate(dataloader),total=len(dataloader),desc="Eval",leave=False):
            out=net(sample_batch["desc1"].to(device=devicegpu),sample_batch["desc2"].to(device=devicegpu))
            out=out.cpu()
            outlabel=out
            label=sample_batch['label']
            label[label<0.9]=0
            label[label>0.1]=1
            pred.extend(outlabel)
            gt.extend(label)
        pred=np.array(pred,dtype='float32')
        gt=np.array(gt,dtype='float32')
        pred=np.nan_to_num(pred)
        precision, recall, pr_thresholds = metrics.precision_recall_curve(gt,pred)
        F1_score = 2 * precision * recall / (precision + recall)
        F1_score = np.nan_to_num(F1_score)
        testaccur = np.max(F1_score)
        print(datatype,"F1:",testaccur)
        if save and testaccur>maxaccur:
            torch.save(net.state_dict(),'./model/model_'+datatype+str(testaccur)+'.pth')
            torch.save(net.state_dict(),'./model/best.pth')
        return testaccur

if __name__=='__main__':
    train()
