import torch
from net import RINet, RINet_attention
from database import evalDataset, evalDataset_kitti360
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from sklearn import metrics
from matplotlib import pyplot as plt
import sys
import time
import argparse
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device=torch.device("cpu")


def eval(seq='0000', model_file="./model/attention/00.ckpt", data_type='kitti360'):
    net = RINet_attention()
    net.load(model_file)
    if data_type == 'kitti':
        test_dataset = evalDataset(seq)
    elif data_type == 'kitti360':
        test_dataset = evalDataset_kitti360(seq)
    net.to(device=device)
    net.eval()
    testdataloader = DataLoader(
        dataset=test_dataset, batch_size=16384, shuffle=False, num_workers=8)
    pred = []
    gt = []
    with torch.no_grad():
        for i_batch, sample_batch in tqdm(enumerate(testdataloader), total=len(testdataloader), desc="Eval seq "+str(seq)):
            out, _ = net(sample_batch["desc1"].to(
                device=device), sample_batch["desc2"].to(device=device))
            outlabel = out.cpu().tolist()
            label = sample_batch['label']
            pred.extend(outlabel)
            gt.extend(label.tolist())
    pred = np.nan_to_num(pred)
    save_db = np.array([pred, gt])
    save_db = save_db.T
    if not os.path.exists('result'):
        os.mkdir('result')
    np.savetxt(os.path.join('result', seq+'.txt'), save_db, "%.4f")
    precision, recall, pr_thresholds = metrics.precision_recall_curve(gt, pred)
    plt.plot(recall, precision, color='darkorange', lw=2, label='P-R curve')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('DL Precision-Recall Curve')
    plt.legend(loc="lower right")
    F1_score = 2 * precision * recall / (precision + recall)
    F1_score = np.nan_to_num(F1_score)
    F1_max_score = np.max(F1_score)
    print("F1:", F1_max_score)
    plt.show()


def fast_eval(seq='00', model_file="./model/attention/00.ckpt", desc_file='./data/desc_kitti/00.npy', pair_file='./data/pairs_kitti/neg_100/00.txt', use_l2_dis=False):
    net = RINet_attention()
    net.load(model_file)
    net.to(device=device)
    net.eval()
    print(net)
    desc_o = np.load(desc_file)/50.0
    descs_torch = torch.from_numpy(desc_o).to(device)
    total_time = 0.
    with torch.no_grad():
        torch.cuda.synchronize()
        time1 = time.time()
        descs = net.gen_feature(descs_torch).cpu().numpy()
        torch.cuda.synchronize()
        total_time += (time.time()-time1)
    print("Feature time:", total_time)
    pairs = np.genfromtxt(pair_file, dtype='int32').reshape(-1, 3)
    if use_l2_dis:
        desc1 = descs[pairs[:, 0]]
        desc2 = descs[pairs[:, 1]]
        time1 = time.time()
        diff = desc1-desc2
        diff = 1./np.sum(diff*diff, axis=1)
        print("Score time:", time.time()-time1)
        diff = diff.reshape(-1, 1)
        diff = np.nan_to_num(diff)
        label = pairs[:, 2].reshape(-1, 1)
        # diff_pos=diff[label>0.9]
        # diff_neg=diff[label<0.2]
        # plt.plot(list(range(len(diff_pos))),diff_pos,'b.')
        # plt.plot(list(range(len(diff_pos),len(diff_pos)+len(diff_neg))),diff_neg,'r.')
        # plt.show()
        precision, recall, pr_thresholds = metrics.precision_recall_curve(
            label, diff)
    else:
        desc1 = torch.from_numpy(descs[pairs[:, 0]]).to(device)
        desc2 = torch.from_numpy(descs[pairs[:, 1]]).to(device)
        total_time = 0
        with torch.no_grad():
            torch.cuda.synchronize()
            time1 = time.time()
            scores, _ = net.gen_score(desc1, desc2)
            scores = scores.cpu().numpy()
            torch.cuda.synchronize()
            total_time += (time.time()-time1)
        print("Score time:", total_time)
        gt = pairs[:, 2].reshape(-1, 1)
        np.savetxt("result/"+seq+'.txt',
                   np.concatenate([scores.reshape(-1, 1), gt.reshape(-1, 1)], axis=1))
        precision, recall, pr_thresholds = metrics.precision_recall_curve(
            gt, scores)
    F1_score = 2 * precision * recall / (precision + recall)
    F1_score = np.nan_to_num(F1_score)
    F1_max_score = np.max(F1_score)
    print("F1:", F1_max_score)
    plt.plot(recall, precision, color='darkorange', lw=2, label='P-R curve')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.show()


def recall(seq='00', model_file="./model/attention/00.ckpt", desc_file='./data/desc_kitti/00.npy', pose_file="./data/pose_kitti/00.txt"):
    poses = np.genfromtxt(pose_file)
    poses = poses[:, [3, 11]]
    inner = 2*np.matmul(poses, poses.T)
    xx = np.sum(poses**2, 1, keepdims=True)
    dis = xx-inner+xx.T
    dis = np.sqrt(np.abs(dis))
    id_pos = np.argwhere(dis <= 5)
    id_pos = id_pos[id_pos[:, 0]-id_pos[:, 1] > 50]
    pos_dict = {}
    for v in id_pos:
        if v[0] in pos_dict.keys():
            pos_dict[v[0]].append(v[1])
        else:
            pos_dict[v[0]] = [v[1]]
    descs = np.load(desc_file)
    descs /= 50.0
    net = RINet_attention()
    net.load(model_file)
    net.to(device=device)
    net.eval()
    # print(net)
    out_save = []
    recall = np.array([0.]*25)
    for v in tqdm(pos_dict.keys()):
        candidates = []
        targets = []
        for c in range(0, v-50):
            candidates.append(descs[c])
            targets.append(descs[v])
        candidates = np.array(candidates, dtype='float32')
        targets = np.array(targets, dtype='float32')
        candidates = torch.from_numpy(candidates)
        targets = torch.from_numpy(targets)
        with torch.no_grad():
            out, _ = net(candidates.to(device=device),
                         targets.to(device=device))
            out = out.cpu().numpy()
            ids = np.argsort(-out)
            o = [v]
            o += ids[:25].tolist()
            out_save.append(o)
            for i in range(25):
                if ids[i] in pos_dict[v]:
                    recall[i:] += 1
                    break
    if not os.path.exists('result'):
        os.mkdir('result')
    np.savetxt(os.path.join('result', seq+'_recall.txt'), out_save, fmt='%d')
    recall /= len(pos_dict.keys())
    print(recall)
    plt.plot(list(range(1, len(recall)+1)), recall, marker='o')
    plt.axis([1, 25, 0, 1])
    plt.xlabel('N top retrievals')
    plt.ylabel('Recall (%)')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', default='08',
                        help='Sequence to eval. [default: 08]')
    parser.add_argument('--dataset', default="kitti",
                        help="Dataset (kitti or kitti360). [default: kitti]")
    parser.add_argument('--model', default="./model/attention/08.ckpt",
                        help='Model file. [default: "./model/attention/08.ckpt"]')
    parser.add_argument('--desc_file', default='./data/desc_kitti/08.npy',
                        help='File of descriptors. [default: ./data/desc_kitti/08.npy]')
    parser.add_argument('--pairs_file', default='./data/pairs_kitti/neg_100/08.txt',
                        help='Candidate pairs. [default: ./data/pairs_kitti/neg_100/08.txt]')
    parser.add_argument('--pose_file', default="./data/pose_kitti/08.txt",
                        help='Pose file (eval_type=recall). [default: ./data/pose_kitti/08.txt]')
    parser.add_argument('--eval_type', default="f1",
                        help='Type of evaluation (f1 or recall). [default: f1]')
    cfg = parser.parse_args()
    if cfg.dataset == "kitti" and cfg.eval_type == "f1":
        fast_eval(seq=cfg.seq, model_file=cfg.model,
                  desc_file=cfg.desc_file, pair_file=cfg.pairs_file)
        # eval(seq=cfg.seq,model_file=cfg.model,data_type=cfg.dataset)
    elif cfg.dataset == "kitti" and cfg.eval_type == "recall":
        recall(cfg.seq, cfg.model, cfg.desc_file, cfg.pose_file)
    elif cfg.dataset == "kitti360" and cfg.eval_type == "f1":
        eval(seq=cfg.seq, model_file=cfg.model, data_type=cfg.dataset)
    else:
        print("Error")
