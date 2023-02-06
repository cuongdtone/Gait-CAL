# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 1/15/2023
import os

import matplotlib.pyplot as plt
import torch
from models.autoencoder import AE
import numpy as np
from datasets_gait.create_vid_dataset import CreateVidDataset
import pickle
from cfg import SETTING

device = torch.device('cpu')
net = AE()
net.load_state_dict(torch.load('src/ae_epoch_19_loss_0.04.pt'))
net.eval()

mode = 'train'

root = r'E:\Gait\GaitDatasetB-silh'
save_root = f'casia_b_phase_2_{mode}_{SETTING}_dataset'

dataset = CreateVidDataset(root, mode=mode)
print(dataset.__len__())

with torch.no_grad():
    for i in range(len(dataset)):
        # print(i)
        vid, path = dataset.__getitem__(i)
        gei = np.zeros((112, 112))
        features = []

        for frame in vid:
            code = net.encoder(frame.unsqueeze(0))
            code = code.view(-1).numpy().tolist()
            features.append(code)
            gei += frame[0].numpy()
        features = np.array(features)

        dir_save_pkl = path.replace(root, save_root)
        os.makedirs(os.path.dirname(dir_save_pkl), exist_ok=True)
        pkl_path = dir_save_pkl + '.pkl'

        # print(pkl_path)
        with open(pkl_path, 'wb') as handle:
            data = {'ae_feat': features, 'gei': gei}
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #
        # print(path)
        # print(features.shape)
        # plt.imshow(gei)
        # plt.show()

        # break