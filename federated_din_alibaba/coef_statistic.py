import os
import pickle
import numpy as np
import tensorflow as tf
import sys
import time
import random
from config import SEND_RECEIVE_CONF as SRC
from model import Model_DIN
from communication import Communication
from data_iterator import DataIterator
import general_functions as gn_fn

with open('./taobao_data_process/taobao_user_item_cate_count.pkl', 'rb') as f:
    user_count, item_count, cate_count = pickle.load(f)

dataset_info = {'user_count': user_count,
                'item_count': item_count,
                'cate_count': cate_count}

total_users_num = dataset_info['user_count']
train_batch_size = 2
user_num = np.zeros(dataset_info['user_count'])
item_num = np.zeros(dataset_info['item_count'])
cate_num = np.zeros(dataset_info['cate_count'])
user_coef = np.ones(dataset_info['user_count'])
item_coef = np.ones(dataset_info['item_count'])
cate_coef = np.ones(dataset_info['cate_count'])
total_set_size = 0

for client in range(1, total_users_num + 1):
    try:
        train_set = DataIterator('./taobao_data_process/taobao_datasets/user_%s' % (str(client)), train_batch_size)
        train_set_size = 0
        user_IDs = []
        item_IDs = []
        cate_IDs = []

        for src, tgt in train_set:
            for example in src:
                user_IDs.append(example[0])
                item_IDs.append(example[1])
                cate_IDs.append(example[2])
                item_IDs += example[3]
                cate_IDs += example[4]
                train_set_size += 1

        user_IDs = list(set(user_IDs))
        item_IDs = list(set(item_IDs))
        cate_IDs = list(set(cate_IDs))
        user_IDs.sort()   
        item_IDs.sort()
        cate_IDs.sort()

        # print("train_set_size: ", train_set_size)

        total_set_size += train_set_size 

        for i in range(len(user_IDs)):
            user_num[user_IDs[i]] += train_set_size

        for i in range(len(item_IDs)):
            item_num[item_IDs[i]] += train_set_size  

        for i in range(len(cate_IDs)):
            cate_num[cate_IDs[i]] += train_set_size
    except:
        pass

for uu in range(dataset_info['user_count']):
    if user_num[uu] > 0:
        user_coef[uu] = total_set_size / user_num[uu]
for ii in range(dataset_info['item_count']):
    if item_num[ii] > 0:
        item_coef[ii] = total_set_size / item_num[ii]
for cc in range(dataset_info['cate_count']):
    if cate_num[cc] > 0:
        cate_coef[cc] = total_set_size / cate_num[cc]

print("user coef: ", len(user_coef))
print("item coef: ", len(item_coef))
print("cate coef: ", len(cate_coef))

for i in range(len(user_coef)):
    with open("coefs/user_coef.txt", 'a') as r:
        r.write(str(user_coef[i]) +"\n")
    r.close()

for i in range(len(item_coef)):
    with open("coefs/item_coef.txt", 'a') as r:
        r.write(str(item_coef[i]) +"\n")
    r.close()


for i in range(len(cate_coef)):
    with open("coefs/cate_coef.txt", 'a') as r:
        r.write(str(cate_coef[i]) +"\n")
    r.close()