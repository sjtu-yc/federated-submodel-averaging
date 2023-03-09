import numpy as np

from config import SEND_RECEIVE_CONF as SRC

from data_iterator import DataIterator


user_count, item_count = 1870, 33136
total_users_num = user_count
train_batch_size = 1
user_num = np.zeros(user_count)
item_num = np.zeros(item_count)

user_coef = np.ones(user_count)
item_coef = np.ones(item_count)

total_set_size = 0

for client in range(1, total_users_num):
    try:
        train_set = DataIterator('./user_data/users/user_%s' % (str(client)), train_batch_size)
        train_set_size = 0
        user_IDs = []
        item_IDs = []

        for src, tgt in train_set:
            for example in src:
                user_IDs.append(example[0])
                item_IDs.append(example[1])
                item_IDs += example[2]

                train_set_size += 1

        user_IDs = list(set(user_IDs))
        item_IDs = list(set(item_IDs))

        user_IDs.sort()   
        item_IDs.sort()

        total_set_size += train_set_size 

        for i in range(len(user_IDs)):
            user_num[user_IDs[i]] += train_set_size

        for i in range(len(item_IDs)):
            item_num[item_IDs[i]] += train_set_size  

    except:
        pass

for uu in range(user_count):
    if user_num[uu] > 0:
        user_coef[uu] = total_set_size / user_num[uu]
for ii in range(item_count):
    if item_num[ii] > 0:
        item_coef[ii] = total_set_size / item_num[ii]

print("user coef: ", len(user_coef))
print("item coef: ", len(item_coef))

for i in range(len(user_coef)):
    with open("coefs/user_coef.txt", 'a') as r:
        r.write(str(user_coef[i]) +"\n")
    r.close()

for i in range(len(item_coef)):
    with open("coefs/item_coef.txt", 'a') as r:
        r.write(str(item_coef[i]) +"\n")
    r.close()
