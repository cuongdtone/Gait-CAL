# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 1/15/2023

import torch
import os
from models.gait_fc import GaitFCV2
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

from create_gallery_nm_1_4 import model, save_root, root, device

# device = torch.device('cpu')
# model = GaitFCV2()
# model.load_state_dict(torch.load('src_mt_setting/128d/gait_fcv2_epoch_1_loss_1.1089105729786854.pt'))
# model.to(device)
# model.eval()

db_root = save_root  # 'gallery_mt_setting_1'
attach_root = root  # 'casia_b_phase_2_eval_dataset'


def calc_euclidean(feat1, feat2):
    feat1 = feat1.ravel()
    feat2 = feat2.ravel()
    return torch.dot(feat1, feat2) / (torch.linalg.norm(feat1) * torch.linalg.norm(feat2))


def evaluation(walking_angle, walking_conditions):
    all_angles = ['000', '018', '036', '054', '072', '090', '108', '126', '144', '162', '180']

    accs = []
    
    for a in all_angles:
        print(a, walking_angle)
        """ GET DB """
        if walking_angle == a:  # pass same angle attach and db
            continue
            
        subjects = os.listdir(db_root)
        db_embedding = []
        db_targets = []
        # print(subjects)
        for subject in subjects:
            subject_angle = os.path.join(db_root, subject, a)
            # print(subject_angle)
            if os.path.exists(subject_angle):
                embedding_nm14 = [os.path.join(subject_angle, em) for em in os.listdir(subject_angle)]
                for emb in embedding_nm14:
                    with open(emb, 'rb') as handle:
                        data = pickle.load(handle)
                    # print(data.shape, subjec)
                    db_embedding.append(data)
                    db_targets.append(subject)

        # print(len(db_targets))
        # print(max([int(i) for i in db_targets]))
        # print(len(set(db_targets)))

        cer_true = []
        cer_pred = []
        truth = []
        pred = []
        false = 0
        true = 0
        """ GET ATTACH """
        subjects = os.listdir(attach_root)
        for subject in subjects:
            for walking_condition in walking_conditions:
                subject_condition = os.path.join(attach_root, subject, walking_condition)
                # print(subject_condition)
                data_attach = os.path.join(subject_condition, walking_angle)
                # print(data_attach)

                # print(data.keys(), subject)

                ################################# GET FEAT FOR ATTACH SUBJECT
                if not os.path.exists(data_attach + '.pkl'):
                    continue
                with open(data_attach + '.pkl', 'rb') as handle:
                    data = pickle.load(handle)
                a_v = data['ae_feat']
                a_gei = np.expand_dims(data['gei'], axis=0) / 40
                a_v = torch.from_numpy(a_v)
                a_gei = torch.from_numpy(a_gei).float()
                a_v = a_v.unsqueeze(0).to(device).float()
                a_gei = a_gei.unsqueeze(0).to(device).float()
                with torch.no_grad():
                    attach_feat = model(a_v, a_gei)
                ####################################

                ################################## ATTACH TO GALLERY
                onehot_true = [0] * (int(max([int(i) for i in db_targets])) + 1)
                onehot_true[int(subject)] = 1
                onehot_pred = [0] * (int(max([int(i) for i in db_targets])) + 1)

                max_d = -1
                predict_subject = -1
                for db_target, db_embed in zip(db_targets, db_embedding):
                    d = calc_euclidean(attach_feat, db_embed)
                    # print(int(db_target) - 63)
                    if d.item() > onehot_pred[int(db_target)]:
                        onehot_pred[int(db_target)] = d.item()

                    if d.item() > max_d:
                        predict_subject = int(db_target)
                        max_d = d.item()

                if predict_subject == int(subject):
                    true += 1
                else:
                    false += 1
                truth.append(int(subject))
                pred.append(predict_subject)

                cer_true.append(onehot_true[63:])
                cer_pred.append(onehot_pred[63:])
                # print('hera')
                # print(onehot_true[63:])
                # print(onehot_pred[63:])
                # print(onehot_true.index(max(onehot_true)), onehot_pred.index(max(onehot_pred)))
            #     break
            # break
        ''' CALCULATE ACCURACY '''

        acc = accuracy_score(truth, pred)
        accs.append(acc)

    print(accs)
    return sum(accs)/len(accs)


if __name__ == '__main__':
    angles = ['000', '018', '036', '054', '072', '090', '108', '126', '144', '162', '180']
    # print(len(angles))
    conditions = [['nm-06', 'nm-05'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']]

    import xlsxwriter

    workbook = xlsxwriter.Workbook('demo.xlsx')
    worksheet = workbook.add_worksheet()

    # acc = evaluation('000', ['nm-06', 'nm-05'])
    # print(acc)
    for idx1, condition in enumerate(conditions):
        print('----' * 10, condition)
        for idx2, angle in enumerate(angles):
            acc = evaluation(angle, condition)
            worksheet.write(idx1 + 1, idx2 + 1, acc * 100)
            print(angle, acc)
    workbook.close()
