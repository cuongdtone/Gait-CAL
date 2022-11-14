# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 10/25/2022


import os
import pickle
import random

import numpy as np
from torch.utils.data import Dataset
from torch.nn import functional as F


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def find_indices(list_to_check, item_to_find, not_equal=False):
    indices = []
    if not_equal:
        for idx, value in enumerate(list_to_check):
            if value != item_to_find:
                indices.append(idx)
    else:
        for idx, value in enumerate(list_to_check):
            if value == item_to_find:
                indices.append(idx)
    return indices


class GaitFCDataset(Dataset):
    def __init__(self, root):
        ids = os.listdir(root)
        self.videos = []
        self.target = []
        for idx in ids:
            videos = os.listdir(os.path.join(root, idx))
            videos = [os.path.join(root, idx, v) for v in videos]
            self.videos += videos
            self.target += [int(idx)] * len(videos)
        # print(self.videos)
        # print(len(self.videos))
        # print(len(self.target))

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, item):
        vid = self.videos[item]
        target = self.target[item]

        positive = find_indices(self.target, target)
        positive.remove(item)
        positive = random.choice(positive)
        positive = self.videos[positive]

        # Negative selection
        negative = find_indices(self.target, target, not_equal=True)
        negative = random.choice(negative)
        negative = self.videos[negative]

        with open(vid, 'rb') as handle:
            anchor_vid = pickle.load(handle)
            a_v = anchor_vid['ae_feat']
            # a_v = softmax(a_v)
            a_gei = anchor_vid['gei']
            a_gei = np.expand_dims(a_gei, axis=0)/40
        with open(positive, 'rb') as handle:
            positive_vid = pickle.load(handle)
            p_v = positive_vid['ae_feat']
            # p_v = softmax(p_v)
            p_gei = positive_vid['gei']
            p_gei = np.expand_dims(p_gei, axis=0)/40
        with open(negative, 'rb') as handle:
            negative_vid = pickle.load(handle)
            n_v = negative_vid['ae_feat']
            # n_v = softmax(n_v)
            n_gei = negative_vid['gei']
            n_gei = np.expand_dims(n_gei, axis=0)/40

        if a_v.shape[0] != 40 or p_v.shape[0] != 40 or n_v.shape[0] != 40:
            return self.__getitem__(random.randint(0, self.__len__() - 1))
        return a_v, a_gei, p_v, p_gei, n_v, n_gei


class GaitFCDatasetV2(Dataset):
    def __init__(self, root):
        ids = os.listdir(root)
        self.videos = []
        self.target = []
        for idx in ids:
            videos = os.listdir(os.path.join(root, idx))
            videos = [os.path.join(root, idx, v) for v in videos]
            self.videos += videos
            self.target += [int(idx)] * len(videos)
        # print(self.videos)
        # print(len(self.videos))
        # print(len(self.target))
        self.means = np.array([np.load('src/means_ae.npy')] * 40)
        self.std = np.array([np.load('src/std_ae.npy')]*40)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, item):
        vid = self.videos[item]
        target = self.target[item]

        with open(vid, 'rb') as handle:
            anchor_vid = pickle.load(handle)
            a_v = anchor_vid['ae_feat']
            # a_v = softmax(a_v)
            a_v = self.norm(a_v)
            a_gei = anchor_vid['gei']
            a_gei = np.expand_dims(a_gei, axis=0)/40
        if a_v.shape[0] != 40:
            return self.__getitem__(random.randint(0, self.__len__() - 1))
        return a_v, a_gei, target

    def norm(self, x):
        return (x - self.means)/self.std



if __name__ == '__main__':
    data = GaitFCDataset(r'E:\Gait\GaitFC\gait_fc_dataset\test')
    print(len(data))
    a_v, a_gei, p_v, p_gei, n_v, n_gei = data.__getitem__(0)
    print(a_v.shape)
    print(a_gei.shape)
    # print(n.shape)
