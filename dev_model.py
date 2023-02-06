# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 1/15/2023
import torch
import pickle


def calc_euclidean(feat1, feat2):
    feat1 = feat1.ravel()
    feat2 = feat2.ravel()
    return torch.dot(feat1, feat2) / (torch.linalg.norm(feat1) * torch.linalg.norm(feat2))


with open('gallery_mt_setting/063/000/nm-01.pkl', 'rb') as handle:
    data1 = pickle.load(handle)

f = []
final = []
for i in range(63, 125):
    try:
        with open('gallery_mt_setting/%03d/180/nm-01.pkl'%i, 'rb') as handle:
            data2 = pickle.load(handle)
        f.append(sum(data2[0][0:128])/128)
        final.append(sum(data2[0][128:])/128)
        # print(sum(data2[0][0:128])/128)
        # print(sum(data2[0][128:])/128)
    except:
        pass

print('Mean of 128 first elements (AL block): ', float(sum(f)/len(f)))
print('Mean of 128 final elements (GE block): ', float(sum(final)/len(final)))
# print(data1)
# print(calc_euclidean(data1, data2))
