# -*- coding: UTF-8 -*-
from data_iterator import DataIterator
import general_functions as gn_fn


for i in range(1, 5):
    train_set = DataIterator('./taobao_datasets/user_%s' % (str(i)), 32, 'train')
    count = 0
    for src, tgt in train_set:
        if count >= 1000:
            break
        print(count)
        count = count + 1
        uids, mids, cats, mid_his, cat_his, mid_mask, target, sl = gn_fn.prepare_data(src, tgt)
        with open('./MNN_train_data/uids_'+str(i)+'.txt', 'a') as f:
            for num in uids:
                print(num, file=f)
        with open('./MNN_train_data/mids_'+str(i)+'.txt', 'a') as f:
            for num in mids:
                print(num, file=f)
        with open('./MNN_train_data/cats_'+str(i)+'.txt', 'a') as f:
            for num in cats:
                print(num, file=f)
        with open('./MNN_train_data/mid_his_'+str(i)+'.txt', 'a') as f:
            for num_line in mid_his:
                for num in num_line:
                    f.write(str(num) + ' ')
                f.write('\n')
        with open('./MNN_train_data/cat_his_'+str(i)+'.txt', 'a') as f:
            for num_line in cat_his:
                for num in num_line:
                    f.write(str(num) + ' ')
                f.write('\n')
        with open('./MNN_train_data/mid_mask_'+str(i)+'.txt', 'a') as f:
            for num_line in mid_mask:
                for num in num_line:
                    f.write(str(num) + ' ')
                f.write('\n')
        with open('./MNN_train_data/target_'+str(i)+'.txt', 'a') as f:
            for num_line in target:
                for num in num_line:
                    f.write(str(num) + ' ')
                f.write('\n')
        with open('./MNN_train_data/sl_'+str(i)+'.txt', 'a') as f:
            for num in sl:
                print(num, file=f)
