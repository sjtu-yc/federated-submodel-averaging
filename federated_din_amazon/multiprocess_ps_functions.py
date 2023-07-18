# -*- coding: UTF-8 -*-
import os
import signal
import multiprocessing
import ssl
import sys
import time
import math
import tensorflow as tf
import numpy as np
import pickle
import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from model import Model_DIN
from communication import Communication
from data_iterator import DataIterator
from copy import deepcopy
from config import SSL_CONF as SC
from config import SEND_RECEIVE_CONF as SRC
from datetime import datetime
import evaluate as my_eval
import general_functions as gn_fn

# set the connection information
PS_PUBLIC_IP = '0.0.0.0:25380'         # Public IP of the ps
PS_PRIVATE_IP = '0.0.0.0:25380'        # Private IP of the ps

# kill all child processes on termination
def my_termination(sig_num, addtion):
    print ('Terminating processes...')
    os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)

class Client:
    def __init__(self, sock, address, ID="-1"):
        self.connection_socket = sock
        self.address = address
        self.ID = ID

def get_sub_model(model_parameters, user_IDs, item_IDs):
    if len(user_IDs) == 1 and len(item_IDs) == 1 and user_IDs[0] == 0 and item_IDs[0] == 0:
        return
    for i, model_parameter in enumerate(model_parameters):
        # print("************************shape**********************")
        # print(i, model_parameter.shape[0])
        if i == 0:
            model_parameters[i] = model_parameter[user_IDs]
        elif i == 1:
            model_parameters[i] = model_parameter[item_IDs]
        else:
            break


def choice_send_parameter(communication, received_dict, timestamp_queue, historical_model_dict, layer_name_queue, connection_socket):
    """Here should keep all the ID list be ascending before do the map to new ID list in build_dataset"""
    user_IDs = received_dict['user_IDs']
    item_IDs = received_dict['item_IDs']

    # print(user_IDs, item_IDs)

    model_timestamp = timestamp_queue[-1]  # get the newest model's timestamp
    model_parameters = deepcopy(historical_model_dict[model_timestamp])   # copy the model
    get_sub_model(model_parameters, user_IDs, item_IDs)

    send_dict = {'model_timestamp': model_timestamp, 'model_paras': model_parameters, 'layer_names': layer_name_queue}
    send_message = gn_fn.create_DINMessage_from_dict(send_dict)
    communication.send_message(send_message, connection_socket)
    print("sending messsage")


def get_info_and_return_model(communication, current_client, batches_info_dict, timestamp_queue, historical_model_dict, layer_name_queue):
    """
    Get information of next batches from the client,
    send back subset of current model to the client according to received message,
    then update batches_info_dict, this will be used when updating model.
    """
    try:
        current_client.connection_socket.send(SRC.please_send_batches_info)
        #received_message = communication.get_np_array(current_client.connection_socket)
        received_message = communication.get_message(current_client.connection_socket)
        received_dict = gn_fn.restore_dict_from_DINMessage(received_message)
        current_client.ID = str(received_dict['client_ID'])
        print(multiprocessing.current_process().name + 'Received next_batches_info from client ' + str(received_dict['client_ID']))
        sys.stdout.flush()

        # get and send back the corresponding embedding and the model parameters
        signal = current_client.connection_socket.recv(10)
        if signal == SRC.please_send_model:
            choice_send_parameter(communication, received_dict, timestamp_queue, historical_model_dict, layer_name_queue, current_client.connection_socket)
            print(multiprocessing.current_process().name + 'Sent back model parameters to client ' + str(received_dict['client_ID']))
            sys.stdout.flush()

        #update batches_info_dict
        batches_info_dict[str(received_dict['client_ID'])] = received_dict
    except Exception as e:
        print(e)
        print(multiprocessing.current_process().name + 'Fallen client: ' + current_client.ID + ' at ' + current_client.address[0] + ':' + str(current_client.address[1]))
        sys.stdout.flush()
        current_client.connection_socket.close()


#============================= START: assistant functions for classify_connections =====================================
def send_hyperparameters(communication, current_client, hyperparameters):
    try:
        # communication.send_np_array(hyperparameters, current_client.connection_socket)
        send_dict = {'hyperparameters': hyperparameters}
        send_message = gn_fn.create_DINMessage_from_dict(send_dict)
        communication.send_message(send_message, current_client.connection_socket)
    except Exception as e:
        print(multiprocessing.current_process().name + e)
        sys.stdout.flush()
        current_client.connection_socket.close()


def accept_new_connections(classification_queue, classification_queue_lock):
    """Called by main thread, keep accepting new connections and append to classification_queue."""
    signal.signal(signal.SIGTERM, my_termination)

    communication = Communication(True, PS_PRIVATE_IP, PS_PUBLIC_IP)
    while True:
        try:
            connection_socket, address = communication.ps_socket.accept()
            """connection_socket = ssl.wrap_socket(
                sock,
                server_side=True,
                certfile=SC.cert_path,
                keyfile=SC.key_path,
                ssl_version=ssl.PROTOCOL_TLSv1)"""

            if classification_queue_lock.acquire():
                classification_queue.append(Client(connection_socket, address))
                classification_queue_lock.release()
            print(multiprocessing.current_process().name + 'Connected from: ' + address[0] + ':' + str(address[1]))
            sys.stdout.flush()

        except Exception as e:
            print(multiprocessing.current_process().name + e)
            sys.stdout.flush()
            continue
#============================== END: assistant functions for classify_connections ======================================


def classify_connections(classification_queue, classification_queue_lock, get_update_queue, get_update_queue_lock,\
                         hyperparameters, batches_info_dict, timestamp_queue, historical_model_dict, layer_name_queue):
    """
    Classify new connections (INIT/UPDATE),
    initialization takes place immediately, using hyperparameters, dataset_info, sess,
    clients willing to send updates will be appended to get_update_queue.
    """
    signal.signal(signal.SIGTERM, my_termination)
    
    communication = Communication(False, PS_PRIVATE_IP, PS_PUBLIC_IP)
    while True:
        if len(classification_queue) == 0:  # Check if there exists connection to be classified
            time.sleep(5)   # Sleep to avoid busy loop
            continue
        elif len(classification_queue) > 0:    # Begin classifying
            if classification_queue_lock.acquire():    # Get lock
                if len(classification_queue) > 0:
                    current_client = classification_queue.pop(0)
                    classification_queue_lock.release()    # Release lock
                else:
                    classification_queue_lock.release()    # Release lock
                    continue
            else:
                continue

            try:    # Communicate to know this connection's purpose
                current_client.connection_socket.settimeout(60)
                current_client.connection_socket.send(SRC.purpose)
                purpose = current_client.connection_socket.recv(10)

                if purpose == SRC.update:   # Append current_client to get_update_queue
                    if get_update_queue_lock.acquire():
                        get_update_queue.append(current_client)
                        get_update_queue_lock.release()
                elif purpose == SRC.init:   # Do initialization
                    send_hyperparameters(communication, current_client, hyperparameters)
                    get_info_and_return_model(communication, current_client, batches_info_dict, timestamp_queue, historical_model_dict, layer_name_queue)
                    print(multiprocessing.current_process().name + 'Initialized client: ' + current_client.ID)
                    sys.stdout.flush()
                    current_client.connection_socket.close()
                    del current_client
                else:
                    print(multiprocessing.current_process().name + 'Unknown purpose: ' + purpose + ', ' + 'purpose should be ' + SRC.init + ' or ' + SRC.update)

            except (OSError, OSError):  # Exception handler
                print(multiprocessing.current_process().name + 'Fallen client: ' + current_client.address[0] + ':' + str(current_client.address[1]))
                sys.stdout.flush()
                try:
                    current_client.connection_socket.close()
                except (OSError, OSError):
                    pass


def get_update(get_update_queue, get_update_queue_lock, return_model_queue, return_model_queue_lock, gathered_weights_dict, gui_event):
    """
    Get updates from clients in get_update_queue and save updates into gathered_weights_dict,
    then pass clients to return_model_queue.
    """
    signal.signal(signal.SIGTERM, my_termination)

    communication = Communication(False, PS_PRIVATE_IP, PS_PUBLIC_IP)
    while True:
        if len(get_update_queue) == 0:  # Check if there exists updates to be gathered
            time.sleep(5)   # Sleep to avoid busy loop
            continue
        elif len(get_update_queue) > 0:
            if get_update_queue_lock.acquire():    # Get lock
                if len(get_update_queue) > 0:
                    current_client = get_update_queue.pop(0)
                    get_update_queue_lock.release()    # Release_lock
                else:
                    get_update_queue_lock.release()    # Release_lock
                    continue
            else:
                continue

            try:    # Communicate to get updates
                current_client.connection_socket.settimeout(60)
                current_client.connection_socket.send(SRC.please_send_update)
                #received_message = communication.get_np_array(current_client.connection_socket)
                received_message = communication.get_message(current_client.connection_socket)
                received_dict = gn_fn.restore_dict_from_DINMessage(received_message)
                current_client.ID = str(received_dict['client_ID'])
                
                if return_model_queue_lock.acquire():   # Get lock
                    return_model_queue.append(current_client)   # Append current_client to return_model_queue
                    gathered_weights_dict[current_client.ID] = received_dict  # Add gathered_weights to gathered_weights_dict
                    return_model_queue_lock.release()   # Release lock
                print(multiprocessing.current_process().name + 'Received updates from client ' + current_client.ID)
                sys.stdout.flush()
                gui_event.set()

            except (OSError, OSError):  # Exception handler
                print(multiprocessing.current_process().name + 'Fallen client: ' + current_client.address[0] + ':' + str(current_client.address[1]))
                sys.stdout.flush()
                try:
                    current_client.connection_socket.close()
                except (OSError, OSError):
                    pass


#================================== START: assistant functions for updating weights ====================================
def delay_compensation(timestamp_queue, historical_model_dict, batches_info_dict, received_dict, delay_compensation_para):
    client_ID = str(received_dict['client_ID'])
    model_timestamp = received_dict['model_timestamp']
    gradients = received_dict['model_paras']
    return 1
    if model_timestamp == timestamp_queue[-1]:
        return 1
    elif model_timestamp in timestamp_queue:   # do delay compensation
        old_model_paras = deepcopy(historical_model_dict[model_timestamp])
        current_model_paras = deepcopy(historical_model_dict[timestamp_queue[-1]])

        user_IDs = batches_info_dict[client_ID]['user_IDs']
        item_IDs = batches_info_dict[client_ID]['item_IDs']

        # get the sub model corresponding to the client's model
        get_sub_model(old_model_paras, user_IDs, item_IDs)
        get_sub_model(current_model_paras, user_IDs, item_IDs)
        gradients = np.array(gradients)
        old_model_paras = np.array(old_model_paras)
        current_model_paras = np.array(current_model_paras)
        # compensated_gradients = gradients + [np.full(weights.shape(), delay_compensation_para) for weights in gradients] *
        # gradients * gradients * (current_model_paras - old_model_paras)
        compensated_gradients = gradients + np.multiply(delay_compensation_para, np.multiply(np.multiply(gradients, gradients), (current_model_paras - old_model_paras)))
        received_dict['model_paras'] = compensated_gradients.tolist()
        print(multiprocessing.current_process().name + 'Client ' + str(client_ID) + '\'s update compensated!')
        sys.stdout.flush()
        return 1
    else:
        print(multiprocessing.current_process().name + 'Client ' + str(client_ID) + '\'s update abandoned, model_timestamp: ' + model_timestamp)
        sys.stdout.flush()
        return 0


def do_update_weights(gathered_weights, placeholders, update_local_vars_op, sess):
    feed_dict = {}
    for place, para in zip(placeholders, gathered_weights):
        feed_dict[place] = para
    sess.run(update_local_vars_op, feed_dict=feed_dict)
    del feed_dict


def gather_delta_model_parameter(clients, clients_message_list, clients_delta_model_list, clients_train_set_size_list,
                                 user_num, item_num, trained_samples_num, size_avg_flag, sess):
    gathered_delta_weights = [np.zeros(weights.shape) for weights in sess.run(tf.trainable_variables())]
    for i in range(len(clients)):
        client_message = clients_message_list[i]
        client_delta_model = clients_delta_model_list[i]
        user_IDs = client_message['user_IDs']
        item_IDs = client_message['item_IDs']

        train_set_size = clients_train_set_size_list[i]
        # aggregate ratio according to one client's training set size, if size_avg_flag
        """if size_avg_flag:   #若size_avg_flag==1则为train_set_size加权
            agg_ratio_i = clients_train_set_size_list[i] * 1.0 / sum(clients_train_set_size_list)
        else:
            agg_ratio_i = 1.0 / len(clients)"""
        if len(user_IDs) == 1 and len(item_IDs) == 1 and user_IDs[0] == 0 and item_IDs[0] == 0: # 手机客户端开发时MNN未支持端上模型动态调整，故特殊处理（ToDo）
            trained_samples_num[0] += train_set_size
            for index, delta_model_parameter in enumerate(client_delta_model):
                if index == 0:
                    for client_index in range(len(delta_model_parameter)):
                        ps_index = client_index
                        gathered_delta_weights[index][ps_index] += delta_model_parameter[client_index] * train_set_size
                        user_num[ps_index] += train_set_size
                elif index == 1:
                    for client_index in range(len(delta_model_parameter)):
                        ps_index = client_index
                        gathered_delta_weights[index][ps_index] += delta_model_parameter[client_index] * train_set_size
                        item_num[ps_index] += train_set_size
                else:
                    gathered_delta_weights[index] = gathered_delta_weights[index] + delta_model_parameter * train_set_size
        else:
            trained_samples_num[0] += train_set_size

            #ps_index: parameter index in the global model
            #client_index: parameter index in the submodel
            for index, delta_model_parameter in enumerate(client_delta_model):
                if index == 0:
                    for client_index in range(len(delta_model_parameter)):
                        ps_index = user_IDs[client_index]
                        gathered_delta_weights[index][ps_index] += delta_model_parameter[client_index] * train_set_size
                        user_num[ps_index] += train_set_size
                elif index == 1:
                    for client_index in range(len(delta_model_parameter)):
                        ps_index = item_IDs[client_index]
                        gathered_delta_weights[index][ps_index] += delta_model_parameter[client_index] * train_set_size
                        item_num[ps_index] += train_set_size
                else:
                    gathered_delta_weights[index] += delta_model_parameter * train_set_size
    return gathered_delta_weights

def do_weights_average(gathered_delta_weights, dataset_info, user_num, item_num, trained_samples_num, sess):
    original_weights = sess.run(tf.trainable_variables())
    new_weights = [np.zeros(weights.shape) for weights in original_weights]
    for index, delta_weights in enumerate(gathered_delta_weights):
        if index == 0:
            for ii in range(dataset_info['user_count']):
                # pay attention to divide zero error
                if user_num[ii] > 0:
                    gathered_delta_weights[index][ii] = gathered_delta_weights[index][ii] / user_num[ii]
                    new_weights[index][ii] = original_weights[index][ii] + gathered_delta_weights[index][ii]
                else:
                    new_weights[index][ii] = original_weights[index][ii]
        elif index == 1:
            for ii in range(dataset_info['item_count']):
                # pay attention to divide zero error
                if item_num[ii] > 0:
                    gathered_delta_weights[index][ii] = gathered_delta_weights[index][ii] / item_num[ii]
                    new_weights[index][ii] = original_weights[index][ii] + gathered_delta_weights[index][ii]
                else:
                    new_weights[index][ii] = original_weights[index][ii]

        else:
            if trained_samples_num[0] > 0:
                gathered_delta_weights[index] = gathered_delta_weights[index] / trained_samples_num[0]
                new_weights[index] = original_weights[index] + gathered_delta_weights[index]
            else:
                new_weights[index] = original_weights[index]
    return new_weights

def fedsubavg_do_weights_average(gathered_delta_weights, dataset_info, trained_samples_num, sess):
    user_coef = np.loadtxt('coefs/user_coef.txt', dtype=str, delimiter="\t")
    item_coef = np.loadtxt('coefs/item_coef.txt', dtype=str, delimiter="\t")

    original_weights = sess.run(tf.trainable_variables())
    new_weights = [np.zeros(weights.shape) for weights in original_weights]
    for index, delta_weights in enumerate(gathered_delta_weights):
        if index == 0:
            for ii in range(dataset_info['user_count']):
                # pay attention to divide zero error
                if trained_samples_num[0] > 0:
                    gathered_delta_weights[index][ii] = float(user_coef[ii]) * gathered_delta_weights[index][ii] / trained_samples_num[0]
                    new_weights[index][ii] = original_weights[index][ii] + gathered_delta_weights[index][ii]
                else:
                    new_weights[index][ii] = original_weights[index][ii]
        elif index == 1:
            for ii in range(dataset_info['item_count']):
                # pay attention to divide zero error
                if trained_samples_num[0] > 0:
                    gathered_delta_weights[index][ii] = float(item_coef[ii]) * gathered_delta_weights[index][ii] / trained_samples_num[0]
                    new_weights[index][ii] = original_weights[index][ii] + gathered_delta_weights[index][ii]
                else:
                    new_weights[index][ii] = original_weights[index][ii]

        else:
            if trained_samples_num[0] > 0:
                gathered_delta_weights[index] = gathered_delta_weights[index] / trained_samples_num[0]
                new_weights[index] = original_weights[index] + gathered_delta_weights[index]
            else:
                new_weights[index] = original_weights[index]
    return new_weights

def fedavg_do_weights_average(gathered_delta_weights, dataset_info, trained_samples_num, sess):
    original_weights = sess.run(tf.trainable_variables())
    new_weights = [np.zeros(weights.shape) for weights in original_weights]
    for index, delta_weights in enumerate(gathered_delta_weights):
        if index == 0:
            for ii in range(dataset_info['user_count']):
                # pay attention to divide zero error
                if trained_samples_num[0] > 0:
                    gathered_delta_weights[index][ii] = gathered_delta_weights[index][ii] / trained_samples_num[0]
                    new_weights[index][ii] = original_weights[index][ii] + gathered_delta_weights[index][ii]
                else:
                    new_weights[index][ii] = original_weights[index][ii]
        elif index == 1:
            for ii in range(dataset_info['item_count']):
                # pay attention to divide zero error
                if trained_samples_num[0] > 0:
                    gathered_delta_weights[index][ii] = gathered_delta_weights[index][ii] / trained_samples_num[0]
                    new_weights[index][ii] = original_weights[index][ii] + gathered_delta_weights[index][ii]
                else:
                    new_weights[index][ii] = original_weights[index][ii]
        else:
            if trained_samples_num[0] > 0:
                gathered_delta_weights[index] = gathered_delta_weights[index] / trained_samples_num[0]
                new_weights[index] = original_weights[index] + gathered_delta_weights[index]
            else:
                new_weights[index] = original_weights[index]
    return new_weights

def save_historical_model(historical_model_dict, timestamp_queue, historical_model_num, g1, sess):
    current_timestamp = str(time.time())
    with g1.as_default():
        current_model = deepcopy(sess.run(tf.trainable_variables()))
    historical_model_dict[current_timestamp] = current_model
    timestamp_queue.append(current_timestamp)
    while len(timestamp_queue) > historical_model_num:  # "if" should work properly here, use "while" in case of unknown exceptions
        oldest_timestamp = timestamp_queue.pop(0)
        historical_model_dict.pop(oldest_timestamp)

def create_csv(path):
    with open(path, 'w') as f:
        csv_writer = csv.writer(f)
        csv_head = ["round_num", "test_auc", "train_auc", "train_loss"]
        csv_writer.writerow(csv_head)

def write_csv(path, round_num, test_auc, train_auc, loss_sum):
    with open(path, 'a+') as f:
        csv_writer = csv.writer(f)
        data_row = [round_num, test_auc, train_auc, loss_sum]
        csv_writer.writerow(data_row)

class Plot:
    def __init__(self):
        self.xs = []
        self.ys = []
        self.round_num = -1
        self.y_range = [0.50,0.51]
        plt.figure('Figure Object 1', figsize=(16, 9), dpi=200)
        plt.ion()
        plt.title('AUC during the training process')
        plt.grid(True)
        #plt.ylim(self.y_range[0], self.y_range[1])
        #plt.yticks(np.linspace(self.y_range[0], self.y_range[1], 11, endpoint=True))

    def add(self, x, y):
        self.xs.append(x)
        self.ys.append(y)
        flag = False
        if (y > self.y_range[1]): 
            self.y_range[1] = y
            flag = True
        self.round_num += 1

        plt.cla()
        plt.xlabel('Round Num')
        plt.ylabel('AUC')
        if (self.round_num == 1):
            pass
        elif (self.round_num <= 19):
            plt.xlim(0, 19)
            plt.xticks(np.linspace(0, 19, 20, endpoint=True))
            plt.ylim(self.y_range[0], self.y_range[1])
            plt.yticks(np.linspace(self.y_range[0], self.y_range[1], 11, endpoint=True))
            plt.plot(self.xs, self.ys, marker = '.')
        elif (self.round_num > 19):
            plt.xlim(0, self.round_num)
            plt.xticks(np.arange(0, self.round_num + 1, step = math.ceil(self.round_num/20.0)))
            plt.ylim(self.y_range[0], self.y_range[1])
            plt.yticks(np.linspace(self.y_range[0], self.y_range[1], 11, endpoint=True))
            plt.plot(self.xs, self.ys, marker = '.')
        
        plt.draw() # plot new figure
        plt.pause(0.1)

        # loc can be [upper, lower, left, right, center]
        # plt.legend(loc="lower right", shadow=False)
        # plt.ioff()

    def savefig(self, path):
        plt.savefig(path)
#=================================== END: assistant functions for updating weights =====================================

def update_model(return_model_queue, return_model_queue_lock, gathered_weights_dict, valid_updates_queue,\
                 return_model_only_queue, return_model_queue_lock2, returning_model_queue, returning_model_queue_lock,\
                 batches_info_dict, timestamp_queue, historical_model_dict, layer_name_queue, hyperparameters,\
                 gui_evaluate_result, gui_event, gui_round_finished):
    """
    Aggregate all updates in this round and apply to the global model
    """
    signal.signal(signal.SIGTERM, my_termination)

    communication = Communication(False, PS_PRIVATE_IP, PS_PUBLIC_IP)
    
    #------------------------------------------ START: build the global model -------------------------------------------
    test_data = DataIterator('./central_data/amazon_test', hyperparameters['test_batch_size'])
    # with open('./taobao_data_process/taobao_item_cate_dict.pkl', 'rb') as f:
    #     item_cate_dict = pickle.load(f)
    # with open('./taobao_data_process/taobao_user_item_cate_count.pkl', 'rb') as f2:
    #     user_count, item_count, cate_count = pickle.load(f2)
    user_count, item_count = 1870, 33136

    dataset_info = {'user_count': user_count,
                    'item_count': item_count,
                    'local_iter_num': hyperparameters['local_iter_num'],
                    'train_batch_size': hyperparameters['train_batch_size']}

    CHECKPOINT_DIR = 'save_path/parameter_sever'
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    CHECKPOINT_DIR = CHECKPOINT_DIR + '/ckpt'

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    model = Model_DIN(user_count, item_count, hyperparameters['embedding_dim'], hyperparameters['opt_alg'])
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    g1 = tf.get_default_graph()
    saver = tf.train.Saver()
    # saver.restore(sess, './save_path_1/parameter_sever/ckpt')
    placeholders = gn_fn.create_placeholders()
    update_local_vars_op = gn_fn.assign_vars(tf.trainable_variables(), placeholders)


    variables_pack_for_eval_and_save = {
        'model': model,
        'saver': saver,
        'test_set': test_data,
        'test_batch_size': hyperparameters['test_batch_size'],
        'best_auc': 0,
        'CHECKPOINT_DIR': CHECKPOINT_DIR,
        'best_round': 0
    }

    g1.finalize()

    layer_name_queue.extend([v.name[0:-2] for v in tf.trainable_variables()])
    historical_model_dict[str(0.01)] = deepcopy(sess.run(tf.trainable_variables()))   # save the initialized model
    timestamp_queue.append(str(0.01))
    #------------------------------------------- END: build the global model --------------------------------------------

    path = "./results.csv"
    create_csv(path)

    last_timestamp = datetime.now()
    round_num = 0
    valid_updates_dict = {}
    # get original model's auc
    # train_set = DataIterator('./central_data/amazon_train', hyperparameters['test_batch_size'])
    # train_auc, loss = my_eval.eval_train_loss(variables_pack_for_eval_and_save, train_set, sess)
    train_auc, loss = 0, 0 
    test_auc, loss_sum, accuracy_sum = my_eval.eval_and_save(variables_pack_for_eval_and_save, round_num, sess)
    print('-----------------------------------------------------------------------')
    print('Initialized model: test_auc: %.4f ---- train loss: %f ---- accuracy: %f' % (test_auc, loss, accuracy_sum))
    print('-----------------------------------------------------------------------')
    sys.stdout.flush()

    #---------------------------------------------------------------------------------------------
    #with g1.as_default():
    #    variable_names = [v.name for v in tf.trainable_variables()]
    #    values = sess.run(variable_names)
    #    for k, v in zip(variable_names, values):
    #        print("Variable: ", k)
    #        print("Shape: ", v.shape)
    #    sys.stdout.flush()
    #---------------------------------------------------------------------------------------------

    write_csv(path, round_num, test_auc, train_auc, loss)

    #-------------------------------------------- START: build the graph ---------------------------------------------
    #my_plot = Plot()
    #my_plot.add(0, test_auc)
    gui_evaluate_result.append([round_num, test_auc, loss])
    gui_event.set()
    gui_round_finished.set()
    #--------------------------------------------- END: build the graph ----------------------------------------------

    round_num = 1

    while True:
        if len(return_model_queue) == 0:
            time.sleep(1)  # Sleep to avoid busy loop
            continue
        elif len(return_model_queue) > 0:    # do delay compensation
            return_model_queue_lock.acquire() # Get lock
            temp_return_model_queue = deepcopy(return_model_queue)
            temp_gathered_weights_dict = deepcopy(gathered_weights_dict)
            return_model_queue[:] = []
            gathered_weights_dict.clear()
            return_model_queue_lock.release() # Release_lock
            for client in temp_return_model_queue:
                received_dict = temp_gathered_weights_dict[client.ID]
                flag = 1
                if flag == 1:
                    valid_updates_dict[client.ID] = received_dict
                    valid_updates_queue.append(client)
                    gui_event.set()
                else:
                    return_model_only_queue.append(client)
            del temp_return_model_queue
            del temp_gathered_weights_dict

        if len(valid_updates_queue) >= hyperparameters['sync_parameter']:
            # preparing data for aggregation
            user_num = np.zeros(dataset_info['user_count'])
            item_num = np.zeros(dataset_info['item_count'])

            trained_samples_num = np.zeros(1)

            clients_message_list = []
            clients_delta_model_list = []
            clients_train_set_size_list = []
            client_loss_list = []
            for current_client in valid_updates_queue:
                # record and split messages from client
                received_dict = valid_updates_dict[current_client.ID]
                client_message = batches_info_dict[current_client.ID]
                client_delta_model = received_dict['model_paras']
                clients_message_list.append(client_message)
                clients_delta_model_list.append(client_delta_model)
                clients_train_set_size_list.append(client_message['client_train_set_size'])
                client_loss = received_dict['local_loss']
                client_loss_list.append(client_loss)

            with g1.as_default():
                # Gather model parameter updates from clients
                """
                federated averaging delta weights controlled by (size_avg_flag=False, adaptive_flag=False)
                Case 1: Default parameters will degenerate to emitting the initial model at parameter server (mathematically proven), 
                                                           and aggregating the delta weights from clients evenly;
                Case 2: (True, False) will aggregate the delta weights from clients according to their training_set_size;
                Case 3: (True, True) will adaptively aggregate the delta weights from client first evenly, 
                                                                                   then according to their training_set_size.
                """
                size_avg_flag = 3
                gathered_delta_weights = gather_delta_model_parameter(valid_updates_queue, clients_message_list, clients_delta_model_list,
                                                                      clients_train_set_size_list, user_num, item_num, trained_samples_num, size_avg_flag, sess)
                
                # print("*****************************************************")
                # do_coef_statistics(gathered_delta_weights, dataset_info, user_num,
                #                                  item_num, trained_samples_num)
                # print("*****************************************************")

                # new_weights = do_weights_average(gathered_delta_weights, dataset_info, user_num,
                #                                  item_num, trained_samples_num, sess)

                new_weights = fedsubavg_do_weights_average(gathered_delta_weights, dataset_info, trained_samples_num, sess)
                
                # Update global model at parameter server
                do_update_weights(new_weights, placeholders, update_local_vars_op, sess)
                print(multiprocessing.current_process().name + 'Round {}: {} seconds consumed.'.format(round_num, (datetime.now() - last_timestamp).seconds))
                last_timestamp = datetime.now()
                print(multiprocessing.current_process().name + 'Round {}: Weights received, average appled '.format(round_num) +
                      'among {} clients'.format(len(valid_updates_queue)) + ', model updated! Evaluating...')
                sys.stdout.flush()

                # Save model into historical_model_dict
                save_historical_model(historical_model_dict, timestamp_queue, hyperparameters['historical_model_num'], g1, sess)

                #------------------------------------- START: add to returning_model_queue --------------------------------------
                return_model_queue_lock2.acquire() # prevent return and heartbeat happen at the same time
                temp_valid_updates_queue = valid_updates_queue[:]
                temp_return_model_only_queue = return_model_only_queue[:]
                valid_updates_queue[:] = []
                valid_updates_dict.clear()
                return_model_only_queue[:] = []
                return_model_queue_lock2.release()
                returning_model_queue_lock.acquire()
                returning_model_queue.extend(temp_valid_updates_queue)
                returning_model_queue.extend(temp_return_model_only_queue)
                returning_model_queue_lock.release()
                #-------------------------------------- END: add to returning_model_queue ---------------------------------------
                gui_round_finished.set()

                # Evaluate global model
                if round_num % 1 == 0:
                    # train_auc, loss = my_eval.eval_train_loss(variables_pack_for_eval_and_save, train_set, sess)
                    test_auc, loss_sum, accuracy_sum = my_eval.eval_and_save(variables_pack_for_eval_and_save,
                                                                             round_num, sess)
                    print(multiprocessing.current_process().name + '-----------------------------------------------------------------------')
                    print(multiprocessing.current_process().name + 'Round %d Model Performance: test_auc: %.4f ---- loss: %f ---- accuracy: %f' %
                          (round_num, test_auc, loss, accuracy_sum))
                    print(multiprocessing.current_process().name + 'Best round: ' + str(variables_pack_for_eval_and_save['best_round']) +
                          ' Best test_auc: ' + str(variables_pack_for_eval_and_save['best_auc']))
                    print(multiprocessing.current_process().name + '-----------------------------------------------------------------------')
                    print('')
                    print('')
                    sys.stdout.flush()
                
                    write_csv(path, round_num, test_auc, train_auc, loss)
                    gui_evaluate_result.append([round_num, test_auc, loss])
                    gui_event.set()
                round_num += 1

            del clients_message_list
            del clients_delta_model_list
            del clients_train_set_size_list
            del temp_valid_updates_queue
            del temp_return_model_only_queue


def return_model(returning_model_queue, returning_model_queue_lock, batches_info_dict, timestamp_queue, historical_model_dict, layer_name_queue, hyperparameters, gui_event):
    """
    For each client in returning_model_queue,
    get_info_and_return_model.
    """
    signal.signal(signal.SIGTERM, my_termination)
    communication = Communication(False, PS_PRIVATE_IP, PS_PUBLIC_IP)

    while True:
        if len(returning_model_queue) == 0:  # Check if there exists updates to be gathered
            time.sleep(5)   # Sleep to avoid busy loop
            continue
        elif len(returning_model_queue) > 0:
            if returning_model_queue_lock.acquire():    # Get lock
                if len(returning_model_queue) > 0:
                    current_client = returning_model_queue.pop(0)
                    returning_model_queue_lock.release()    # Release_lock
                else:
                    returning_model_queue_lock.release()    # Release_lock
                    continue
            else:
                continue

            try:    # Communicate to get updates
                get_info_and_return_model(communication, current_client, batches_info_dict, timestamp_queue, historical_model_dict, layer_name_queue)
                current_client.connection_socket.close()
                gui_event.set()
                #print(multiprocessing.current_process().name + 'Returned the global model to client ' + current_client.ID)
                #sys.stdout.flush()

            except (OSError, OSError):  # Exception handler
                print(multiprocessing.current_process().name + 'Fallen client: ' + current_client.address[0] + ':' + str(current_client.address[1]))
                sys.stdout.flush()
                try:
                    current_client.connection_socket.close()
                except (OSError, OSError):
                    pass

#================================= START: assistant functions for heartbeat ============================================
def check_heartbeat(current_client):
    try:
        current_client.connection_socket.settimeout(20)
        current_client.connection_socket.send(SRC.heartbeat)
        heartbeat = current_client.connection_socket.recv(10)
        if heartbeat == SRC.heartbeat:
            return True
        else:
            return False
    except (OSError, OSError):
        return False
#================================== END: assistant functions for heartbeat =============================================


def get_update_queue_heartbeat(get_update_queue, get_update_queue_lock, interval_time):
    """
    This function keeps connections to clients in get_update_queue alive by sending and reciving heartbeats.
    """
    signal.signal(signal.SIGTERM, my_termination)
    while True:
        time.sleep(interval_time)  # Sleep for interval_time seconds
        if len(get_update_queue) <= 1:  # Check if there exists updates to be gathered
            time.sleep(interval_time)   # Sleep for interval_time seconds
            continue
        elif len(get_update_queue) > 1:
            print(multiprocessing.current_process().name + 'Sending heartbeat for get_update_queue...')
            sys.stdout.flush()
            if get_update_queue_lock.acquire(): # Get lock
                temp_get_update_queue = get_update_queue[:]    # Copy the queue
                get_update_queue_lock.release() # Release_lock
            else:
                continue

            temp_get_update_queue.pop(0)    # Skip the first client since it may immediately be connected by get_update function
                                            # Unless the get_update function is stuck for more than 120s, this operation shouldn't cause the first client being timeout
            for current_client in temp_get_update_queue:
                flag = check_heartbeat(current_client)
                if flag == 0:
                    print(multiprocessing.current_process().name + 'Fallen client: ' + current_client.address[0] + ':' + str(current_client.address[1]))
                    sys.stdout.flush()
                    try:
                        if get_update_queue_lock.acquire():  # Get lock
                            try:
                                get_update_queue.remove(current_client) # Delete this client from queue
                            except Exception:
                                pass
                            get_update_queue_lock.release()  # Release_lock
                        current_client.connection_socket.close()    # Close connection
                    except (OSError, OSError):
                        pass

            print(multiprocessing.current_process().name + 'Heartbeat check for get_update_queue finished!')
            sys.stdout.flush()
            del temp_get_update_queue


def return_model_queue_heartbeat(return_model_queue, return_model_queue_lock, gathered_weights_dict, valid_updates_queue, return_model_only_queue,\
                                 return_model_queue_lock2, batches_info_dict, interval_time):
    """
    This function keeps connections to clients in return_model_queue alive by sending and reciving heartbeats.
    """
    signal.signal(signal.SIGTERM, my_termination)
    while True:
        time.sleep(interval_time)  # Sleep for interval_time seconds
        if len(return_model_queue) + len(valid_updates_queue) + len(return_model_only_queue) > 0:
            print(multiprocessing.current_process().name + 'Sending heartbeat for return_model_queue...')
            sys.stdout.flush()
            if return_model_queue_lock.acquire():   # Get lock
                temp_return_model_queue = return_model_queue[:]    # Copy the queue
                return_model_queue_lock.release()   # Release_lock
            else:
                continue
            if return_model_queue_lock2.acquire():  # Get lock
                temp_valid_updates_queue = valid_updates_queue[:]
                temp_return_model_only_queue = return_model_only_queue[:]
                return_model_queue_lock2.release()  # Release_lock
            else:
                continue

            temp_return_model_queue = temp_return_model_queue + temp_valid_updates_queue + temp_return_model_only_queue
            for current_client in temp_return_model_queue:
                flag = check_heartbeat(current_client)
                if flag == 0:
                    try:
                        if return_model_queue_lock.acquire():  # Get lock
                            print(multiprocessing.current_process().name + 'Fallen client: ' + current_client.ID)
                            sys.stdout.flush()
                            try:
                                return_model_queue.remove(current_client)   # Delete this client from queue
                                gathered_weights_dict.pop(current_client.ID)    # Also delete its data
                                batches_info_dict.pop(current_client.ID)
                            except Exception:
                                pass
                            return_model_queue_lock.release()  # Release_lock
                        current_client.connection_socket.close()    # Close connection
                    except (OSError, OSError):
                        pass
            print(multiprocessing.current_process().name + 'Heartbeat check for return_model_queue finished!')
            sys.stdout.flush()
            del temp_return_model_queue
            del temp_valid_updates_queue
            del temp_return_model_only_queue

def returning_model_queue_heartbeat(returning_model_queue, returning_model_queue_lock, interval_time):
    """
    This function keeps connections to clients in return_model_queue alive by sending and reciving heartbeats.
    """
    signal.signal(signal.SIGTERM, my_termination)
    while True:
        time.sleep(interval_time)  # Sleep for interval_time seconds
        if len(returning_model_queue) <= 1:  # Check if there exists updates to be gathered
            time.sleep(interval_time)   # Sleep for interval_time seconds
            continue
        elif len(returning_model_queue) > 1:
            print(multiprocessing.current_process().name + 'Sending heartbeat for returning_model_queue...')
            sys.stdout.flush()
            if returning_model_queue_lock.acquire(): # Get lock
                temp_returning_model_queue = returning_model_queue[:]    # Copy the queue
                returning_model_queue_lock.release() # Release_lock
            else:
                continue

            temp_returning_model_queue.pop(0)    # Skip the first client since it may immediately be connected by get_update function
                                            # Unless the get_update function is stuck for more than 120s, this operation shouldn't cause the first client being timeout
            for current_client in temp_returning_model_queue:
                flag = check_heartbeat(current_client)
                if flag == 0:
                    print(multiprocessing.current_process().name + 'Fallen client: ' + current_client.address[0] + ':' + str(current_client.address[1]))
                    sys.stdout.flush()
                    try:
                        if returning_model_queue_lock.acquire():  # Get lock
                            try:
                                returning_model_queue.remove(current_client) # Delete this client from queue
                            except Exception:
                                pass
                            returning_model_queue_lock.release()  # Release_lock
                        current_client.connection_socket.close()    # Close connection
                    except (OSError, OSError):
                        pass

            print(multiprocessing.current_process().name + 'Heartbeat check for returning_model_queue finished!')
            sys.stdout.flush()
            del temp_returning_model_queue