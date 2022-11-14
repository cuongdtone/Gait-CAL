# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 8/24/2022
import matplotlib.pyplot as plt

from models.autoencoder import AE
from tqdm import tqdm
import cv2
import numpy as np
import glob
import os
import random
import torch
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
import pickle
from datasets.video_dataset import VideoDataset


def save_new_pickle(data, save_dir):
    c = 0
    while os.path.exists(os.path.join(save_dir, '%03d.pkl'%(c))):
        c += 1
    path = os.path.join(save_dir, '%03d.pkl'%(c))
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


device = torch.device('cpu')
net = AE()
net.load_state_dict(torch.load('src/ae_epoch_19_loss_0.04.pt'))
net.eval()

dataset = VideoDataset(r'E:\Gait\dataset', txt_list_file='src/test.txt', img_size=112, vid_len=40)
print(f'Dataset have {len(dataset)} video')
save_dataset = 'dataset/test'
os.makedirs(save_dataset, exist_ok=True)
#  this dir contain code of gait with: 1 person 1 folder, folder contain pickles file of

with torch.no_grad():
    for i in tqdm(range(len(dataset))):
        vid, target = dataset.__getitem__(i)
        # print(target)
        features = []
        for frame in vid:
            code = net.encoder(frame.unsqueeze(0))
            code = code.view(-1).numpy().tolist()
            features.append(code)
        features = np.array(features)
        path_save = os.path.join(save_dataset, str(target))
        os.makedirs(path_save, exist_ok=True)
        save_new_pickle(features, path_save)

        # with open(path_save + '/000.pkl', 'rb') as handle:
        #     anchor_vid = pickle.load(handle)
        # print(anchor_vid.shape)
        # break



