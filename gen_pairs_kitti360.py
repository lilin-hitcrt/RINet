import os
import numpy as np
from tqdm import tqdm
import json
import random
from operator import itemgetter


def gen_pairs(sequ, neg_num=1):
    folder = "/media/l/yp2/KITTI-360/labels/2013_05_28_drive_" + sequ+"_sync"
    pose_file = "/media/l/yp2/KITTI-360/data_poses/2013_05_28_drive_"+sequ+"_sync/poses.txt"
    label_files = os.listdir(folder)
    label_files.sort()
    indexs = [int(v.split(".")[0]) for v in label_files]
    posedata = np.genfromtxt(pose_file)
    pose_indexs = posedata[:, 0]
    pose_indexs = [int(v) for v in pose_indexs]
    pose = posedata[:, 1:].reshape(-1, 3, 4)[:, 0:2, 3].tolist()
    pose_dict = dict(zip(pose_indexs, pose))
    pose_valid = itemgetter(*indexs)(pose_dict)
    pose_valid = np.array(pose_valid)
    inner = 2*np.matmul(pose_valid, pose_valid.T)
    xx = np.sum(pose_valid**2, 1, keepdims=True)
    dis = xx-inner+xx.T
    dis = np.sqrt(np.abs(dis))
    score = 1.-1./(1+np.exp((10.-dis)/1.5))
    id = np.argwhere(dis > -1)
    id = id[id[:, 0] >= id[:, 1]]
    label = score[(id[:, 0], id[:, 1])]
    label = label.reshape(-1, 1)
    indexs = np.array(indexs, dtype='int')
    id[:, 0] = indexs[id[:, 0]]
    id[:, 1] = indexs[id[:, 1]]
    out = np.concatenate((id, label), 1)
    out_pos = out[out[:, 2] > 0.1]
    out_neg = out[out[:, 2] <= 0.1]
    print(out_pos.shape)
    print(out_neg.shape)
    np.savez(sequ+'.npz', pos=out_pos, neg=out_neg)


if __name__ == '__main__':
    sequs = ['0000', '0002', '0003', '0004',
             '0005', '0006', '0007', '0009', '0010']
    for sequ in tqdm(sequs):
        gen_pairs(sequ, 10)
