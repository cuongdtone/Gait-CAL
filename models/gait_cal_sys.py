# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 10/31/2022
import os

import cv2
import matplotlib.pyplot as plt
import torch
from models.autoencoder import AE
from models.gait_fc import GaitFCV2
from torchvision import transforms
from PIL import Image
import numpy as np


class GaitEncoding:
    def __init__(self):

        self.device = torch.device('cpu')
        self.ae = AE()
        self.ae.load_state_dict(torch.load('src/ae_epoch_19_loss_0.04.pt'))
        self.ae.eval()

        transform_list = [transforms.Grayscale(1),
                          transforms.Resize(112),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5,), (0.5,))]
        self.transform = transforms.Compose(transform_list)

        self.model = GaitFCV2()
        self.model.load_state_dict(torch.load('src/gait_fcv2_epoch_27.pt'))
        self.model.eval()

    def __call__(self, clip):
        ae_feats = []
        gei = np.zeros((112, 112))
        for frame in clip:
            mask = crop(frame)
            mask = Image.fromarray(mask)
            mask = self.transform(mask)
            # plt.imshow(mask[0])
            # plt.show()
            with torch.no_grad():
                code = self.ae.encoder(mask.unsqueeze(0))
            code = code.view(-1).numpy().tolist()
            ae_feats.append(code)
            gei += mask[0].numpy()
        ae_feats = np.array(ae_feats)
        features = self.softmax(ae_feats)
        gei = np.expand_dims(gei, axis=0)/40

        a_v = torch.from_numpy(features).float()
        a_gei = torch.from_numpy(gei).float()

        # print(a_gei.max(), a_gei.min())
        # print(a_v)
        a_v = a_v.unsqueeze(0).to(self.device)
        a_gei = a_gei.unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(a_v, a_gei)
        return feat

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)


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




