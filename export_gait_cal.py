# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 10/31/2022


import os
import pickle
import random
import torch
from models.gait_fc import GaitFC, GaitFCV2
import numpy as np

device = torch.device('cpu')
model = GaitFCV2()
model.load_state_dict(torch.load('src/gait_fcv2_epoch_27.pt'))

model.to(device)
model.eval()

x_ae = torch.rand(1, 40, 324)
x_gei = torch.rand(1, 1, 112, 112)

y = model(x_ae, x_gei)
print(y.shape)

torch.onnx.export(model, (x_ae, x_gei), 'src/gait_cal.onnx', export_params=True, opset_version=12, do_constant_folding=True,
                  input_names = ['al', 'ge'],
                  output_names = ['output'],
                  dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                'output': {0: 'batch_size'}}
                  )
