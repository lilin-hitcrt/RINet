import string
import torch
from net import RINet, RINet_attention
from database import evalDataset_kitti360, SigmoidDataset_kitti360, SigmoidDataset_train, SigmoidDataset_eval
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import metrics
import os
import argparse
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard.writer import SummaryWriter
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(cfg):
    writer = SummaryWriter()
    net = RINet_attention()
    net.to(device=device)
    print(net)
    sequs = cfg.all_seqs
    sequs.remove(cfg.seq)
    train_dataset = SigmoidDataset_train(sequs=sequs, neg_ratio=cfg.neg_ratio,
                                         eva_ratio=cfg.eval_ratio, desc_folder=cfg.desc_folder, gt_folder=cfg.gt_folder)
    test_dataset = SigmoidDataset_eval(sequs=sequs, neg_ratio=cfg.neg_ratio,
                                       eva_ratio=cfg.eval_ratio, desc_folder=cfg.desc_folder, gt_folder=cfg.gt_folder)
    # train_dataset=SigmoidDataset_kitti360(['0009','0003','0007','0002','0004','0006','0010'],1)
    # test_dataset=evalDataset_kitti360('0005')
    batch_size = cfg.batch_size
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=6)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters(
    )), lr=cfg.learning_rate, weight_decay=1e-6)
    epoch = cfg.max_epoch
    starting_epoch = 0
    batch_num = 0
    if not cfg.model == "":
        checkpoint = torch.load(cfg.model)
        starting_epoch = checkpoint['epoch']
        batch_num = checkpoint['batch_num']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    for i in range(starting_epoch, epoch):
        net.train()
        pred = []
        gt = []
        for i_batch, sample_batch in tqdm(enumerate(train_loader), total=len(train_loader), desc='Train epoch '+str(i), leave=False):
            optimizer.zero_grad()
            out, diff = net(sample_batch["desc1"].to(
                device=device), sample_batch["desc2"].to(device=device))
            labels = sample_batch["label"].to(device=device)
            loss1 = torch.nn.functional.binary_cross_entropy_with_logits(
                out, labels)
            loss2 = labels*diff*diff+(1-labels)*torch.nn.functional.relu(
                cfg.margin-diff)*torch.nn.functional.relu(cfg.margin-diff)
            loss2 = torch.mean(loss2)
            loss = loss1+loss2
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                writer.add_scalar(
                    'total loss', loss.cpu().item(), global_step=batch_num)
                writer.add_scalar('loss1', loss1.cpu().item(),
                                  global_step=batch_num)
                writer.add_scalar('loss2', loss2.cpu().item(),
                                  global_step=batch_num)
                batch_num += 1
                outlabel = out.cpu().numpy()
                label = sample_batch['label'].cpu().numpy()
                mask = (label > 0.9906840407) | (label < 0.0012710163)
                label = label[mask]
                label[label < 0.5] = 0
                label[label > 0.5] = 1
                pred.extend(outlabel[mask].tolist())
                gt.extend(label.tolist())
        pred = np.array(pred, dtype='float32')
        pred = np.nan_to_num(pred)
        gt = np.array(gt, dtype='float32')
        precision, recall, _ = metrics.precision_recall_curve(gt, pred)
        F1_score = 2 * precision * recall / (precision + recall)
        F1_score = np.nan_to_num(F1_score)
        trainaccur = np.max(F1_score)
        print('Train F1:', trainaccur)
        writer.add_scalar('train f1', trainaccur, global_step=i)
        lastaccur = test(net=net, dataloader=test_loader)
        writer.add_scalar('eval f1', lastaccur, global_step=i)
        print('Eval F1:', lastaccur)
        torch.save({'epoch': i, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(
        ), 'batch_num': batch_num}, os.path.join(cfg.log_dir, cfg.seq, str(i)+'.ckpt'))


def test(net, dataloader):
    net.eval()
    pred = []
    gt = []
    with torch.no_grad():
        for i_batch, sample_batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Eval", leave=False):
            out, _ = net(sample_batch["desc1"].to(
                device=device), sample_batch["desc2"].to(device=device))
            out = out.cpu()
            outlabel = out
            label = sample_batch['label']
            mask = (label > 0.9906840407) | (label < 0.0012710163)
            label = label[mask]
            label[label < 0.5] = 0
            label[label > 0.5] = 1
            pred.extend(outlabel[mask])
            gt.extend(label)
        pred = np.array(pred, dtype='float32')
        gt = np.array(gt, dtype='float32')
        pred = np.nan_to_num(pred)
        precision, recall, pr_thresholds = metrics.precision_recall_curve(
            gt, pred)
        F1_score = 2 * precision * recall / (precision + recall)
        F1_score = np.nan_to_num(F1_score)
        testaccur = np.max(F1_score)
        return testaccur


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='log/',
                        help='Log dir. [default: log]')
    parser.add_argument('--seq', default='00',
                        help='Sequence to test. [default: 00]')
    parser.add_argument('--all_seqs', type=list, default=['00', '01', '02', '03', '04', '05', '06', '07', '08',
                        '09', '10'], help="All sequence. [default: ['00','01','02','03','04','05','06','07','08','09','10'] ]")
    parser.add_argument('--neg_ratio', type=float, default=1,
                        help='The proportion of negative samples used during training. [default: 1]')
    parser.add_argument('--eval_ratio', type=float, default=0.1,
                        help='Proportion of samples used for validation. [default: 0.1]')
    parser.add_argument('--desc_folder', default="./data/desc_kitti",
                        help='Folder containing descriptors. [default: ./data/desc_kitti]')
    parser.add_argument('--gt_folder', default="./data/gt_kitti",
                        help='Folder containing gt files. [default: ./data/gt_kitti]')
    parser.add_argument('--model', default="",
                        help='Pretrained model. [default: ""]')
    parser.add_argument('--max_epoch', type=int, default=20,
                        help='Epoch to run. [default: 20]')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch Size during training. [default: 1024]')
    parser.add_argument('--learning_rate', type=float, default=0.02,
                        help='Initial learning rate. [default: 0.02]')
    parser.add_argument('--weight_decay', type=float,
                        default=1e-6, help='Weight decay. [default: 1e-6]')
    parser.add_argument('--margin', type=float, default=0.2,
                        help='Margin used in contrastive loss. [default: 0.2]')
    cfg = parser.parse_args()
    if(not os.path.exists(os.path.join(cfg.log_dir, cfg.seq))):
        os.makedirs(os.path.join(cfg.log_dir, cfg.seq))
    train(cfg)
