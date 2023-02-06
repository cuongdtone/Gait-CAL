# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 1/15/2023

"""
# Root Contain nm#1-4
    subject_id_0
                0
                18
                36
                54
                72
                ...
                180
    subject_id_...

"""
import torch
import os
from models.gait_fc import GaitFCV2
import pickle
import numpy as np
from cfg import SETTING

device = torch.device('cpu')
model = GaitFCV2()
model.load_state_dict(torch.load(f'src/128d/gait_fcv2_MT_epoch_5_loss_0.005613383910958729.pt'))
model.to(device)
model.eval()


root = 'casia_b_phase_2_eval_dataset'
save_root = f'gallery_{SETTING}_setting_128d_e5_nc400_m0.1'

gallery_conditions = ['nm-01', 'nm-02', 'nm-03', 'nm-04']
subjects = os.listdir(root)
print(len(subjects))


if __name__ == '__main__':
    for subject in subjects:

        for condition in gallery_conditions:
            path_to_conditon = os.path.join(root, subject, condition)
            # print(path_to_conditon)
            views = os.listdir(path_to_conditon)
            for view in views:
                view_angle = view.split('.')[0]
                path_dir_save_feat = os.path.join(save_root, subject, view_angle)
                os.makedirs(path_dir_save_feat, exist_ok=True)
                path_save_feat = os.path.join(path_dir_save_feat, condition + '.pkl')
                path_input_data = os.path.join(path_to_conditon, view)

                with open(path_input_data, 'rb') as handle:
                    data = pickle.load(handle)
                a_v = data['ae_feat']
                a_gei = np.expand_dims(data['gei'], axis=0) / 40

                a_v = torch.from_numpy(a_v)
                a_gei = torch.from_numpy(a_gei).float()

                a_v = a_v.unsqueeze(0).to(device).float()
                a_gei = a_gei.unsqueeze(0).to(device).float()

                print(path_input_data)

                with torch.no_grad():
                    feat = model(a_v, a_gei)

                with open(path_save_feat, 'wb') as handle:
                    pickle.dump(feat, handle, protocol=pickle.HIGHEST_PROTOCOL)

                # print(feat.shape)
                #
                # print(a_gei.shape)
                # print(a_v.shape)
                # print(path_input_data)
                # print(path_save_feat)
                # print()
        # break