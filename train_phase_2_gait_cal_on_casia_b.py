# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 1/15/2023

from datasets_gait.gait_fc_dataset import GaitP2CasiaBDataset
from models.gait_fc import GaitFCV2
import torch
from losses.arcloss import ArcFace, DistCrossEntropy
from torch.nn import functional as F
from cfg import SETTING

train_dataset = GaitP2CasiaBDataset(fr'casia_b_phase_2_train_{SETTING}_dataset')
print(len(train_dataset))

batch_size = 128
epochs = 100
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
device = torch.device('cpu')
model = GaitFCV2()
model.load_state_dict(torch.load('src/gait_fcv2_epoch_27.pt'))
# print(model)
model.to(device)
model.train()

print(len(set(train_dataset.targets)))
arc = ArcFace(256, 400, s=32, m=0.1)

criteron = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

start_epoch = 0
print('start training !')

for epoch in range(start_epoch, epochs):
    model.train()
    losses = []
    iter = 0
    for batch_idx, (a_v, a_gei, target) in enumerate(train_loader):
        a_v = a_v.to(device).float()
        a_gei = a_gei.to(device).float()
        optimizer.zero_grad()
        anchor_out = model(a_v, a_gei)
        # print(target)
        anchor_out = arc(anchor_out, target)
        loss = criteron(anchor_out, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f'Epoch: [%d]/[%d] Training, %.2f%%,{iter}/{len(train_dataset)}, Loss={loss.item()}' % (epoch, epochs, iter*100/len(train_dataset)))
        iter += len(a_v)
    scheduler.step()
    torch.save(model.state_dict(), f'src/128d/gait_fcv2_{SETTING}_epoch_{epoch}_loss_{sum(losses)/len(losses)}.pt')
