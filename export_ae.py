# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 10/31/2022


import torch

from models.autoencoder import AE

device = torch.device('cpu')
net = AE()
net.load_state_dict(torch.load('src/ae_epoch_19_loss_0.04.pt'))
net.eval()

x = torch.rand((1, 112, 112))
y = net(x)

torch.onnx.export(net, x, 'src/ae_onnx.onnx', export_params=True, opset_version=12, do_constant_folding=True,
                  input_names = ['input'],
                  output_names = ['output'],
                  dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                'output': {0: 'batch_size'}}
                  )