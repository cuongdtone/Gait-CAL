# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 10/24/2022

from torch.utils.data import Dataset
from torchvision import transforms
from glob import glob
import os
import cv2
import numpy as np
from PIL import Image
import random


class CustomDataset(Dataset):
    def __init__(self, root, txt_list_file=None, image_size=112, return_target=False):
        print(root)
        self.return_target = return_target
        if txt_list_file is not None:
            self.images = []
            self.target = []
            with open(txt_list_file, 'r', encoding='utf8') as f:
                ids = f.readlines()
                for i in ids:
                    view = os.listdir(root + '/' + i.strip())
                    for v in view:
                        images = os.listdir(root + '/' + i.strip() + '/' + v)
                        images = [os.path.join(root, i.strip(), v, img) for img in images]
                        target = [int(i.strip())] * len(images)
                        self.target += target
                        self.images += images

        else:
            self.images = glob(os.path.join(root, '*/*/*'))

        # print(len(self.images))
        # print(len(self.target))

        transform_list = [transforms.Grayscale(1),
                          transforms.Resize(image_size),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5,), (0.5,))]
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            path = self.images[idx]
            print(path)
            target = self.target[idx]
            image = cv2.imread(path, 0)
            image = self.crop(image)
            image = Image.fromarray(image)
            image = self.transform(image)
            # x = image.view(-1, self.args.img_size[0] ** 2)
        except:
            return self.__getitem__(random.randint(0, self.__len__() - 1))
        return image if self.return_target is False else image, target

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


if __name__ == '__main__':
    dataset = CustomDataset(r'E:\Gait\dataset', r'E:\Gait\GaitFC\src\test.txt')
    print(dataset.__len__())
    # for i in range(len(dataset)):
    #     image = dataset.__getitem__(i)
    #     print(image.shape)
    #     break

