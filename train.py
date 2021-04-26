import torch
from net import RINet,RINet_attention
from database import evalDataset,SigmoidDataset1,evalDataset_kitti360,SigmoidDataset_kitti360
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import  metrics
from tensorboardX import SummaryWriter
def train():
    writer = SummaryWriter('runs/exp')
    device=torch.device('cuda')
    net=RINet_attention()
    # net.load('/home/l/workspace/python/test/model/08/model_test0.9675291730086251.pth')
    net.to(device=device)
    devicecpu=torch.device('cpu')
    print(net)
    # train_dataset=SigmoidDataset1(['07','01','06','03','04','05','02','00','09','10'],1)
    # test_dataset=evalDataset('08')
    train_dataset=SigmoidDataset_kitti360(['0000','0003','0009','0002','0004','0007','0005','0010'],1)
    test_dataset=evalDataset_kitti360('0006')
    batch_size=4096
    train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=8)
    test_loader=DataLoader(dataset=test_dataset,batch_size=4096,shuffle=False,num_workers=8)
    optimizer=torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr=0.02,weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[10,20,30,40,50,60],gamma=0.5)
    # scheduler=torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr=1e-3,max_lr=1,step_size_up=2000,cycle_momentum=False)
    epoch=1000
    maxaccur=0.
    batch_num=0
    for i in range(epoch):
        net.train()
        pred=[]
        gt=[]
        # epoch_loss=0
        batch_loss=0
        for i_batch,sample_batch in tqdm(enumerate(train_loader),total=len(train_loader),desc='Train epoch '+str(i)):
            optimizer.zero_grad()
            out=net(sample_batch["desc1"].to(device=device),sample_batch["desc2"].to(device=device))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(out, sample_batch["label"].to(device=device))
            loss.backward()
            optimizer.step()
            # scheduler.step()
            # print("Loss:",loss)
            with torch.no_grad():
                batch_loss=loss.to(device=devicecpu).item()
                # epoch_loss+=batch_loss
                writer.add_scalar('train loss',batch_loss,global_step=batch_num)
                batch_num+=1
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
        pred=np.nan_to_num(pred)
        gt=np.array(gt,dtype='float32')
        # epoch_loss/=len(pred)
        # print("Epoch ",i," loss:",epoch_loss)
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
        pred=np.nan_to_num(pred)
        precision, recall, pr_thresholds = metrics.precision_recall_curve(gt,pred)
        F1_score = 2 * precision * recall / (precision + recall)
        F1_score = np.nan_to_num(F1_score)
        testaccur = np.max(F1_score)
        print(datatype,"F1:",testaccur)
        if save and testaccur>maxaccur:
            torch.save(net.state_dict(),'./model/model_'+datatype+str(testaccur)+'.pth')
        return testaccur

if __name__=='__main__':
    train()