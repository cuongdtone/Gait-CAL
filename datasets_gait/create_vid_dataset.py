# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 10/24/2022
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
from glob import glob
import os
import cv2
import numpy as np
from PIL import Image
import random


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


class CreateVidDataset(Dataset):
    def __init__(self, root, image_size=112, mode='train'):
        # print(root)
        subject_range = [1, 75] if mode == 'train' else [75, 125]
        self.vid_len = 40
        self.videos = []
        for i in range(*subject_range):
            subject = root + r'\%03d' % i
            print(subject)
            conditions = os.listdir(subject)
            for condition in conditions:
                self.videos += [os.path.join(subject, condition, view) for view in os.listdir(os.path.join(subject, condition))]
        # print(len(self.images))
        # print(len(self.target))

        transform_list = [transforms.Grayscale(1),
                          transforms.Resize(image_size),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5,), (0.5,))]
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        try:
            path = self.videos[idx]
            vid = self.read_frame(path)
        except:
            return self.__getitem__(random.randint(0, self.__len__() - 1))
        return vid, path

    @staticmethod
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


if __name__ == '__main__':
    dataset = CreateVidDataset(r'E:\Gait\GaitDatasetB-silh')
    print(dataset.__len__())
    for i in range(len(dataset)):
        # print(i)
        vid, path = dataset.__getitem__(i)
        for frame in vid:
            plt.imshow(frame[0])
            plt.show()
        # print(path)
        break
    #     print(image.shape)
    #     break
    # print(list(range(1, 63)))

