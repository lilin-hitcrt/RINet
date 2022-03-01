import numpy as np
from matplotlib import pyplot as plt
import sys
import os


def run(seq='00'):
    pose_file = "/media/l/yp2/KITTI/odometry/dataset/poses/"+seq+".txt"
    poses = np.genfromtxt(pose_file)
    poses = poses[:, [3, 11]]
    inner = 2*np.matmul(poses, poses.T)
    xx = np.sum(poses**2, 1, keepdims=True)
    dis = xx-inner+xx.T
    dis = np.sqrt(np.abs(dis))
    id_pos = np.argwhere(dis < 3)
    id_neg = np.argwhere(dis > 20)
    # id_pos=id_pos[id_pos[:,0]-id_pos[:,1]>50]
    id_neg = id_neg[id_neg[:, 0] > id_neg[:, 1]]
    id_pos = np.concatenate(
        [id_pos, (id_pos[:, 0]*0+1).reshape(-1, 1)], axis=1)
    id_neg = np.concatenate([id_neg, (id_neg[:, 0]*0).reshape(-1, 1)], axis=1)
    print(id_pos.shape)
    np.savez(seq+'.npz', pos=id_pos, neg=id_neg)


def run_sigmoid(seq='00'):
    pose_file = "/media/l/yp2/KITTI/odometry/dataset/poses/"+seq+".txt"
    poses = np.genfromtxt(pose_file)
    poses = poses[:, [3, 11]]
    inner = 2*np.matmul(poses, poses.T)
    xx = np.sum(poses**2, 1, keepdims=True)
    dis = xx-inner+xx.T
    dis = np.sqrt(np.abs(dis))
    score = 1.-1./(1+np.exp((10.-dis)/1.5))
    score[dis < 3] = 1
    # plt.imshow(score)
    # plt.show()
    id = np.argwhere(dis > -1)
    id = id[id[:, 0] >= id[:, 1]]
    label = score[(id[:, 0], id[:, 1])]
    label = label.reshape(-1, 1)
    out = np.concatenate((id, label), 1)
    out_pos = out[out[:, 2] > 0.1]
    out_neg = out[out[:, 2] <= 0.1]
    print(out_pos.shape)
    print(out_neg.shape)
    np.savez(seq+'.npz', pos=out_pos, neg=out_neg)


if __name__ == '__main__':
    seq = "00"
    if len(sys.argv) > 1:
        seq = sys.argv[1]
    run_sigmoid(seq)
    # run(seq)
