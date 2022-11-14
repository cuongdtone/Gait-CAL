# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 10/24/2022
import torch
import numpy as np
import matplotlib.pyplot as plt

from models.autoencoder import AE
from datasets.ae_dataset import CustomDataset

model = AE()
model.load_state_dict(torch.load('src/ae_epoch_19_loss_0.04.pt'))
model.eval()

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

dataset = CustomDataset(r'E:\Gait\dataset', txt_list_file='src/train.txt')

with torch.no_grad():
    for i in range(len(dataset)):
        image, _ = dataset.__getitem__(i)
        print(image.shape)
        y = model(image.unsqueeze(0)).detach().cpu().numpy()
        print(softmax(y))
        # plt.imshow(image[0])
        # plt.show()
        # plt.imshow(y[0][0])
        # plt.show()

        break
