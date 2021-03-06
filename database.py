from torch.utils.data import Dataset
import torch
import os
import numpy as np
import random
from matplotlib import pyplot as plt
import json
import random


class SigmoidDataset_eval(Dataset):
    def __init__(self, sequs=['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10'], neg_ratio=1, desc_folder="./data/desc_kitti", gt_folder="./data/gt_kitti", eva_ratio=0.1) -> None:
        super().__init__()
        print(sequs)
        self.descs = []
        self.gt_pos = []
        self.gt_neg = []
        self.pos_nums = [0]
        self.neg_num = 0
        self.pos_num = 0
        for seq in sequs:
            desc_file = os.path.join(desc_folder, seq+'.npy')
            gt_file = os.path.join(gt_folder, seq+'.npz')
            self.descs.append(np.load(desc_file))
            gt = np.load(gt_file)
            pos = gt['pos'][-int(len(gt['pos'])*eva_ratio):]
            neg = gt['neg'][-int(len(gt['neg'])*eva_ratio):]
            self.gt_pos.append(pos)
            self.gt_neg.append(neg)
            self.pos_num += len(self.gt_pos[-1])
            self.pos_nums.append(self.pos_num)
        self.neg_num = int(neg_ratio*self.pos_num)

    def __len__(self):
        return self.pos_num+self.neg_num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        pair = [-1, -1, 0]
        if idx >= self.pos_num:
            id_seq = random.randint(0, len(self.gt_neg)-1)
            id = random.randint(0, len(self.gt_neg[id_seq])-1)
            pair = self.gt_neg[int(id_seq)][id]
            out = {"desc1": self.descs[int(id_seq)][int(
                pair[0])]/50., "desc2": self.descs[int(id_seq)][int(pair[1])]/50., 'label': pair[2]}
            return out
        for i in range(1, len(self.pos_nums)):
            if self.pos_nums[i] > idx:
                pair = self.gt_pos[i-1][idx-self.pos_nums[i-1]]
                out = {"desc1": self.descs[i-1][int(
                    pair[0])]/50., "desc2": self.descs[i-1][int(pair[1])]/50., 'label': pair[2]}
                return out


class SigmoidDataset_train(Dataset):
    def __init__(self, sequs=['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10'], neg_ratio=1, desc_folder="./data/desc_kitti", gt_folder="./data/gt_kitti", eva_ratio=0.1) -> None:
        super().__init__()
        print(sequs)
        self.descs = []
        self.gt_pos = []
        self.gt_neg = []
        self.pos_nums = [0]
        self.neg_num = 0
        self.pos_num = 0
        for seq in sequs:
            desc_file = os.path.join(desc_folder, seq+'.npy')
            gt_file = os.path.join(gt_folder, seq+'.npz')
            self.descs.append(np.load(desc_file))
            gt = np.load(gt_file)
            pos = gt['pos'][:-int(len(gt['pos'])*eva_ratio)]
            neg = gt['neg'][:-int(len(gt['neg'])*eva_ratio)]
            self.gt_pos.append(pos)
            self.gt_neg.append(neg)
            self.pos_num += len(self.gt_pos[-1])
            self.pos_nums.append(self.pos_num)
        self.neg_num = int(neg_ratio*self.pos_num)

    def __len__(self):
        return self.pos_num+self.neg_num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        pair = [-1, -1, 0]
        if idx >= self.pos_num:
            id_seq = random.randint(0, len(self.gt_neg)-1)
            id = random.randint(0, len(self.gt_neg[id_seq])-1)
            pair = self.gt_neg[int(id_seq)][id]
            out = {"desc1": self.descs[int(id_seq)][int(
                pair[0])]/50., "desc2": self.descs[int(id_seq)][int(pair[1])]/50., 'label': pair[2]}
            if random.randint(0, 1) > 0:
                self.rand_occ(out["desc1"])
                self.rand_occ(out["desc2"])
            return out
        for i in range(1, len(self.pos_nums)):
            if self.pos_nums[i] > idx:
                pair = self.gt_pos[i-1][idx-self.pos_nums[i-1]]
                out = {"desc1": self.descs[i-1][int(
                    pair[0])]/50., "desc2": self.descs[i-1][int(pair[1])]/50., 'label': pair[2]}
                if random.randint(0, 1) > 0:
                    self.rand_occ(out["desc1"])
                    self.rand_occ(out["desc2"])
                return out

    def rand_occ(self, in_desc):
        n = random.randint(0, 60)
        s = random.randint(0, 360-n)
        in_desc[:, s:s+n] *= 0


class SigmoidDataset(Dataset):
    def __init__(self, sequs=['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10'], neg_ratio=1, desc_folder="./data/desc_kitti", gt_folder="./data/gt_kitti") -> None:
        super().__init__()
        print(sequs)
        self.descs = []
        self.gt_pos = []
        self.gt_neg = []
        self.pos_nums = [0]
        self.neg_num = 0
        self.pos_num = 0
        for seq in sequs:
            desc_file = os.path.join(desc_folder, seq+'.npy')
            gt_file = os.path.join(gt_folder, seq+'.npz')
            self.descs.append(np.load(desc_file))
            gt = np.load(gt_file)
            self.gt_pos.append(gt['pos'])
            self.gt_neg.append(gt['neg'])
            self.pos_num += len(self.gt_pos[-1])
            self.pos_nums.append(self.pos_num)
        self.neg_num = int(neg_ratio*self.pos_num)

    def __len__(self):
        return self.pos_num+self.neg_num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        pair = [-1, -1, 0]
        if idx >= self.pos_num:
            id_seq = random.randint(0, len(self.gt_neg)-1)
            id = random.randint(0, len(self.gt_neg[id_seq])-1)
            pair = self.gt_neg[int(id_seq)][id]
            out = {"desc1": self.descs[int(id_seq)][int(
                pair[0])]/50., "desc2": self.descs[int(id_seq)][int(pair[1])]/50., 'label': pair[2]*1.}
            if random.randint(0, 2) > 1:
                self.rand_occ(out["desc1"])
                self.rand_occ(out["desc2"])
            return out
        for i in range(1, len(self.pos_nums)):
            if self.pos_nums[i] > idx:
                pair = self.gt_pos[i-1][idx-self.pos_nums[i-1]]
                out = {"desc1": self.descs[i-1][int(pair[0])]/50., "desc2": self.descs[i-1][int(
                    pair[1])]/50., 'label': pair[2]*1.}
                if random.randint(0, 2) > 1:
                    self.rand_occ(out["desc1"])
                    self.rand_occ(out["desc2"])
                return out

    def rand_occ(self, in_desc):
        n = random.randint(0, 60)
        s = random.randint(0, 360-n)
        in_desc[:, s:s+n] *= 0


class evalDataset(Dataset):
    def __init__(self, seq="00", desc_folder="./data/desc_kitti", gt_folder="./data/pairs_kitti/neg_100") -> None:
        super().__init__()
        self.descs = []
        self.pairs = []
        self.num = 0
        desc_file = os.path.join(desc_folder, seq+'.npy')
        pair_file = os.path.join(gt_folder, seq+'.txt')
        self.descs = np.load(desc_file)
        self.pairs = np.genfromtxt(pair_file, dtype='int32')
        self.num = len(self.pairs)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        pair = self.pairs[idx]
        out = {"desc1": self.descs[int(
            pair[0])]/50., "desc2": self.descs[int(pair[1])]/50., 'label': pair[2]}
        angle1 = np.random.randint(0, 359)
        angle2 = np.random.randint(0, 359)
        out["desc1"] = np.roll(out["desc1"], angle1, axis=1)
        out["desc2"] = np.roll(out["desc2"], angle2, axis=1)
        return out


class SigmoidDataset_kitti360(Dataset):
    def __init__(self, sequs=['0000', '0002', '0003', '0004', '0005', '0006', '0007', '0009', '0010'], neg_ratio=1, desc_folder="./data/desc_kitti360", gt_folder="./data/gt_kitti360") -> None:
        super().__init__()
        print(sequs)
        self.descs = []
        self.gt_pos = []
        self.gt_neg = []
        self.key_map = []
        self.pos_nums = [0]
        self.neg_num = 0
        self.pos_num = 0
        for seq in sequs:
            desc_file = os.path.join(desc_folder, seq+'.npy')
            gt_file = os.path.join(gt_folder, seq+'.npz')
            self.descs.append(np.load(desc_file))
            self.key_map.append(
                json.load(open(os.path.join(desc_folder, seq+'.json'))))
            gt = np.load(gt_file)
            self.gt_pos.append(gt['pos'])
            self.gt_neg.append(gt['neg'])
            self.pos_num += len(self.gt_pos[-1])
            self.pos_nums.append(self.pos_num)
        self.neg_num = int(neg_ratio*self.pos_num)

    def __len__(self):
        return self.pos_num+self.neg_num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        pair = [-1, -1, 0]
        if idx >= self.pos_num:
            id_seq = random.randint(0, len(self.gt_neg)-1)
            id = random.randint(0, len(self.gt_neg[id_seq])-1)
            pair = self.gt_neg[int(id_seq)][id]
            out = {"desc1": self.descs[int(id_seq)][self.key_map[int(id_seq)][str(int(
                pair[0]))]]/50., "desc2": self.descs[int(id_seq)][self.key_map[int(id_seq)][str(int(pair[1]))]]/50., 'label': pair[2]}
            if random.randint(0, 1) > 0:
                self.rand_occ(out["desc1"])
                self.rand_occ(out["desc2"])
            return out
        for i in range(1, len(self.pos_nums)):
            if self.pos_nums[i] > idx:
                pair = self.gt_pos[i-1][idx-self.pos_nums[i-1]]
                out = {"desc1": self.descs[i-1][self.key_map[i-1][str(int(
                    pair[0]))]]/50., "desc2": self.descs[i-1][self.key_map[i-1][str(int(pair[1]))]]/50., 'label': pair[2]}
                if random.randint(0, 1) > 0:
                    self.rand_occ(out["desc1"])
                    self.rand_occ(out["desc2"])
                return out

    def rand_occ(self, in_desc):
        n = random.randint(0, 60)
        s = random.randint(0, 360-n)
        in_desc[:, s:s+n] *= 0


class evalDataset_kitti360(Dataset):
    def __init__(self, seq="0000", desc_folder="./data/desc_kitti360", gt_folder="./data/pairs_kitti360/neg10") -> None:
        super().__init__()
        self.descs = []
        self.pairs = []
        self.num = 0
        desc_file = os.path.join(desc_folder, seq+'.npy')
        pair_file = os.path.join(gt_folder, seq+'.txt')
        self.descs = np.load(desc_file)
        self.pairs = np.genfromtxt(pair_file, dtype='int32')
        self.key_map = json.load(open(os.path.join(desc_folder, seq+'.json')))
        self.num = len(self.pairs)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        pair = self.pairs[idx]
        out = {"desc1": self.descs[self.key_map[str(int(
            pair[0]))]]/50., "desc2": self.descs[self.key_map[str(int(pair[1]))]]/50., 'label': pair[2]}
        return out


if __name__ == '__main__':
    database = SigmoidDataset_train(
        ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10'], 2)
    print(len(database))
    for i in range(0, len(database)):
        idx = random.randint(0, len(database)-1)
        d = database[idx]
        print(i, d['label'])
        plt.subplot(2, 1, 1)
        plt.imshow(d['desc1'])
        plt.subplot(2, 1, 2)
        plt.imshow(d['desc2'])
        plt.show()
