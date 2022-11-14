# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 10/24/2022

import os
import random

list_ids = os.listdir(r'E:\Gait\dataset')
random.shuffle(list_ids)

train = list_ids[:400]
test = list_ids[400:]

with open('src/train.txt', 'w', encoding='utf8') as f:
    f.write('\n'.join(train))

with open('src/test.txt', 'w', encoding='utf8') as f:
    f.write('\n'.join(test))