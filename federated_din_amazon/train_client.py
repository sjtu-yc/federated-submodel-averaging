# -*- coding: UTF-8 -*-

import os
import pickle
import numpy as np
import tensorflow as tf
import sys
import time
import random
from config import SEND_RECEIVE_CONF as SRC
from client_model import Model_DIN
from communication import Communication
from data_iterator import DataIterator
import general_functions as gn_fn

flags = tf.app.flags
flags.DEFINE_boolean('is_ps', False, 'True if it is parameter server')
flags.DEFINE_integer('index_num', 1, 'Index of client')
FLAGS = flags.FLAGS

#os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.index_num % 2)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

np.random.seed(1234)
tf.set_random_seed(1234)

PS_PUBLIC_IP = 'localhost:25380'  # Public IP of the ps
PS_PRIVATE_IP = 'localhost:25380'  # Private IP of the ps

'''
CHECKPOINT_DIR = 'save_path/worker_%s' % (str(FLAGS.index_num))  # 根据客户端index产生不同储存目录
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
CHECKPOINT_DIR = CHECKPOINT_DIR + '/ckpt'
'''

# Create the communication object and get the training hyperparameters
communication = Communication(FLAGS.is_ps, PS_PRIVATE_IP, PS_PUBLIC_IP)

client_socket = communication.start_socket_client()
print('Waiting for PS\'s command...')
sys.stdout.flush()
client_socket.settimeout(300)
while True:
    signal = client_socket.recv(10)
    if signal == SRC.purpose:
        print('Sending init purpose...')
        sys.stdout.flush()
        client_socket.send(SRC.init)
        break
    elif signal == SRC.heartbeat:
        print('Sending heartbeat...')
        sys.stdout.flush()
        client_socket.send(SRC.heartbeat)
        continue
    else:
        client_socket.close()
        print('Server Error! Exit!')
        exit(-1)

#hyperparameters = communication.get_np_array(client_socket)
received_message = communication.get_message(client_socket)
received_dict = gn_fn.restore_dict_from_DINMessage(received_message)
hyperparameters = received_dict['hyperparameters']
communication_rounds = hyperparameters['communication_rounds']
local_epoch_num = hyperparameters['local_iter_num'] # permanantly used here, ToDo
train_batch_size = hyperparameters['train_batch_size']
test_batch_size = hyperparameters['test_batch_size']
predict_batch_size = hyperparameters['predict_batch_size']
predict_users_num = hyperparameters['predict_users_num']
predict_ads_num = hyperparameters['predict_ads_num']
learning_rate = hyperparameters['learning_rate']
decay_rate = hyperparameters['decay_rate']
embedding_dim = hyperparameters['embedding_dim']
opt_alg = 'sgd'
total_users_num = 1870

old_model_paras = None
model_timestamp = None

for round_num in range(communication_rounds):
    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    while True:
        train_file = './user_data/users/user_%s' % (str(random.randint(1,total_users_num)))
        if os.path.exists(train_file):
            break
    train_set = DataIterator(train_file, train_batch_size)
    # print("train batch size:", train_batch_size)
    train_set_size = 0

    # generate the training set and the communication ID in this round
    client_training_batches = []
    user_IDs = []
    item_IDs = []

    user_IDs_appear_times_dict = {}
    item_IDs_appear_times_dict = {}


    for src, tgt in train_set:
        # uids, mids, cats, mid_his, cat_his, mid_mask, target, sl = gn_fn.prepare_data(src, tgt)
        # client_training_batches.append([uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, learning_rate])
        client_training_batches.append([src, tgt])
        for example in src:
            user_IDs.append(example[0])
            item_IDs.append(example[1])
            item_IDs += example[2]

            train_set_size += 1
            # the following part is used for counting the number of user, item, cate
            tmpUID = []
            tmpIID = []
            tmpUID.append(example[0])
            tmpIID.append(example[1])
            tmpIID += example[2]

            tmpUID = list(set(tmpUID))
            tmpIID = list(set(tmpIID))

            for UID in tmpUID:
                if UID not in user_IDs_appear_times_dict:
                    user_IDs_appear_times_dict[UID] = 0
                user_IDs_appear_times_dict[UID] += 1
            for IID in tmpIID:
                if IID not in item_IDs_appear_times_dict:
                    item_IDs_appear_times_dict[IID] = 0
                item_IDs_appear_times_dict[IID] += 1
            # the above part is used for counting the number of user, item, cate

    #train_set_size += train_batch_size*local_iter_num

    user_IDs = list(set(user_IDs))
    item_IDs = list(set(item_IDs))

    user_IDs.sort()   # keep the id is ascending, can make the map convenient
    item_IDs.sort()


    user_IDs_appear_times = []
    item_IDs_appear_times = []


    for user_ID in user_IDs:
        user_IDs_appear_times.append(user_IDs_appear_times_dict[user_ID])
    for item_ID in item_IDs:
        item_IDs_appear_times.append(item_IDs_appear_times_dict[item_ID])

    user_IDs_count = len(user_IDs)
    item_IDs_count = len(item_IDs)

    user_IDs_map_old2new = dict(zip(user_IDs, range(user_IDs_count)))
    item_IDs_map_old2new = dict(zip(item_IDs, range(item_IDs_count)))


    for ind, round_training_set in enumerate(client_training_batches):  # [src, tgt]
        for index, value in enumerate(round_training_set[0]):
            value[0] = user_IDs_map_old2new[value[0]]
            value[1] = item_IDs_map_old2new[value[1]]

            for sub_ind, sub_val in enumerate(value[2]):
                value[2][sub_ind] = item_IDs_map_old2new[sub_val]


    # communicate with ps, send batches_info and receive current model
    # client_socket = communication.start_socket_client()
    print('Waiting for PS\'s command...')
    sys.stdout.flush()
    client_socket.settimeout(300)
    while True:
        signal = client_socket.recv(10)
        if signal == SRC.please_send_batches_info:
            print('Sending client data IDs...')
            sys.stdout.flush()
            send_dict = {
                         'user_IDs': user_IDs,
                         'item_IDs': item_IDs,
                         'user_IDs_appear_times': user_IDs_appear_times,
                         'item_IDs_appear_times': item_IDs_appear_times,
                         'client_ID': FLAGS.index_num,
                         'client_train_set_size': train_set_size
                        }
            #communication.send_np_array(send_message, client_socket)
            send_message = gn_fn.create_DINMessage_from_dict(send_dict)
            communication.send_message(send_message, client_socket)
            print('Sending operation over. Receiving corresponding embedding and model parameters...')
            sys.stdout.flush()
            client_socket.send(SRC.please_send_model)
            received_message = communication.get_message(client_socket)
            received_dict = gn_fn.restore_dict_from_DINMessage(received_message)
            # print(received_dict)
            old_model_paras = received_dict['model_paras']
            model_timestamp = received_dict['model_timestamp']
            layer_names = received_dict['layer_names']
            client_socket.close()
            print('Received model parameter.')
            sys.stdout.flush()
            break
        elif signal == SRC.heartbeat:
            print('Sending heartbeat...')
            sys.stdout.flush()
            client_socket.send(SRC.heartbeat)
            continue
        else:
            client_socket.close()
            print('Server Error! Exit!')
            exit(-1)

    # create model and initialize it with the embedding and model parameters pulled from ps
    model = Model_DIN(user_IDs_count, item_IDs_count, embedding_dim, opt_alg)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    placeholders = gn_fn.create_placeholders()
    feed_dict = {}
    for place, para in zip(placeholders, old_model_paras):
        feed_dict[place] = para
    update_local_vars_op = gn_fn.assign_vars(tf.trainable_variables(), placeholders)
    sess.run(update_local_vars_op, feed_dict=feed_dict)
    print('Weights succesfully initialized')
    sys.stdout.flush()

    # begin training process
    # heartbeat_keeper_flag = 1
    print('Begin training')
    sys.stdout.flush()
    start_time = time.time()

    loss_sum = 0.0
    accuracy_sum = 0.
    local_iter_num = 0
    max_iter_num = 10

    while True:
        if local_iter_num >= max_iter_num:
            break
        for src_tgt in client_training_batches:
            uids, mids, mid_his, mid_mask, target, sl = gn_fn.prepare_data(src_tgt[0], src_tgt[1])
            model.get_prox_loss(sess, old_model_paras)
            loss, acc = model.train(sess, [uids, mids, mid_his, mid_mask, target, sl, learning_rate])
            accuracy_sum += acc
            loss_sum += loss
            local_iter_num += 1
            if local_iter_num >= max_iter_num:
                break

    print('%d round training over' % (round_num + 1))
    print('time: %d ----> iter: %d ----> train_loss: %.4f ---- train_accuracy: %.4f' %
          (time.time() - start_time, local_iter_num, loss_sum / local_iter_num, accuracy_sum / local_iter_num))
    print('')
    sys.stdout.flush()

    # preparing update message, delta_model_paras is a list of numpy arrays
    new_model_paras = sess.run(tf.trainable_variables())
    delta_model_paras = [np.zeros(weights.shape) for weights in new_model_paras]
    for index in range(len(new_model_paras)):
        delta_model_paras[index] = new_model_paras[index] - old_model_paras[index]

    send_dict = {'client_ID': FLAGS.index_num, 'model_timestamp': model_timestamp,
                 'model_paras': delta_model_paras, 'layer_names': layer_names,
                 'local_loss': loss_sum/local_iter_num}
    # update learning rate
    learning_rate *= decay_rate

    # connect to ps
    client_socket = communication.start_socket_client()
    client_socket.settimeout(300)
    while True:
        signal = client_socket.recv(10)
        if signal == SRC.purpose:
            client_socket.send(SRC.update)
            break
        elif signal == SRC.heartbeat:
            print('Sending heartbeat...')
            sys.stdout.flush()
            client_socket.send(SRC.heartbeat)
            continue
        else:
            client_socket.close()
            print('Server Error! Exit!')
            exit(-1)

    # send updates
    client_socket.settimeout(300)
    while True:
        signal = client_socket.recv(10)
        if signal == SRC.please_send_update:
            #communication.send_np_array(send_message, client_socket)
            send_message = gn_fn.create_DINMessage_from_dict(send_dict)
            communication.send_message(send_message, client_socket)
            print('Sent trained weights')
            sys.stdout.flush()
            break
        elif signal == SRC.heartbeat:
            print('Sending heartbeat...')
            sys.stdout.flush()
            client_socket.send(SRC.heartbeat)
            continue
        else:
            client_socket.close()
            print('Server Error! Exit!')
            exit(-1)

    print("Client %d trains over in round %d  and takes %f second\n" % (send_dict['client_ID'], \
                                                                        round_num+1, time.time() - start_time))
    print('-----------------------------------------------------------------')
    print('')
    print('')
    sys.stdout.flush()
    sess.close()

print('finished!')
sys.stdout.flush()
