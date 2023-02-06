# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 10/25/2022

from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import numpy as np
import glob
import os
import random
import torch
from PIL import Image


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


def crop(mask):
    x, y, w, h = cv2.boundingRect(mask)
    crop = mask[y:y + h, x:x + w]
    if h > w:
        bg = np.zeros((h, h))
        st_x = int(h / 2 - w / 2)
        ed_x = int(h / 2 + w / 2)
        bg[:, st_x:ed_x] = crop
    else:
        bg = np.zeros((w, w))
        st_y = int(w / 2 - h / 2)
        ed_y = int(w / 2 + h / 2)
        bg[st_y:ed_y, :] = crop
    return bg


class VideoDataset(Dataset):
    def __init__(self, root, txt_list_file, img_size, vid_len):
        self.vid_len = vid_len
        self.root_dir = root

        with open(txt_list_file, 'r', encoding='utf8') as f:
            ids = f.readlines()
            self.people = [i.strip() for i in ids]
        self.targets = list()
        self.videos = list()
        for idx, name in enumerate(self.people):
            path = os.path.join(self.root_dir, name)
            list_video = os.listdir(path)
            for i in list_video:
                self.videos.append(os.path.join(path, i))
                self.targets.append(idx)

        temp = list(zip(self.videos, self.targets))
        random.shuffle(temp)
        res1, res2 = zip(*temp)
        self.videos, self.targets = list(res1), list(res2)

        transform_list = [
                          transforms.Grayscale(1),
                          transforms.Resize(img_size),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5,), (0.5,))]
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        try:
            anchor = self.videos[idx]
            target = self.targets[idx]
            anchor_vid = self.read_frame(anchor)
        except:
            return self.__getitem__(random.randint(0, self.__len__() - 1))
        return anchor_vid, target

    def read_frame(self, path):
        vid_path = os.listdir(path)
        vid_path.sort()
        end = random.randint(self.vid_len + 1, len(vid_path) - 2)

        vid_path = vid_path[(end - self.vid_len):end]
        vid = []
        for p in vid_path:
            try:
                mask = cv2.imread(os.path.join(path, p), 0)
                mask = crop(mask)
                mask = Image.fromarray(mask)
                mask = self.transform(mask)
                vid.append(mask)
            except Exception as e:
                # print(e)
                pass
        return vid
