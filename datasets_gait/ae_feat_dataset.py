# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 10/25/2022


import os
import pickle
import random

from torch.utils.data import Dataset


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


class FeatDataset(Dataset):
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
        with open(positive, 'rb') as handle:
            positive_vid = pickle.load(handle)
        with open(negative, 'rb') as handle:
            negative_vid = pickle.load(handle)
        if anchor_vid.shape[0] != 40 or positive_vid.shape[0] != 40 or negative_vid.shape[0] != 40:
            return self.__getitem__(random.randint(0, self.__len__() - 1))
        return anchor_vid, positive_vid, negative_vid


if __name__ == '__main__':
    data = FeatDataset(r'E:\Gait\GaitFC\dataset\train')
    a, p, n = data.__getitem__(0)
    print(a.shape)
    print(p.shape)
    print(n.shape)
