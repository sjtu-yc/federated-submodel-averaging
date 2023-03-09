# -*- coding: UTF-8 -*-

import os
import signal
import multiprocessing
import sys
import tensorflow as tf
import numpy as np
from copy import deepcopy
from communication import Communication
import multiprocess_ps_functions as ps_fn
import PSGUI.ps_gui as ps_gui
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
np.random.seed(1234)
tf.set_random_seed(1234)

# set the connection information
PS_PUBLIC_IP  = '0.0.0.0:25380'         # Public IP of the ps
PS_PRIVATE_IP = '0.0.0.0:25380'        # Private IP of the ps

# set the number of processes to launch
classifier_num = 4
receiver_num   = 4
# note: the number of updater must be 1 as the global model shouldn't be updated by multiple processes at the same time
returner_num   = 4
# note: usually there is no need to launch multiple heartbeaters for a queue

# training parameters setup
opt_alg = 'sgd'
if opt_alg == 'adam':
    learning_rate = 0.001
    decay_rate = 1.0  # decay rate of learning rate in each communication round
# default is sgd
else:
    learning_rate = 0.1
    decay_rate = 0.999 # decay rate of learning rate in each communication round

hyperparameters = {'communication_rounds': 201,         # number of communication rounds
                   'local_iter_num': 1,                  # now used for the number of local epochs before one round average/communication
                   'train_batch_size': 4,
                   'test_batch_size': 1024,
                   'predict_batch_size': 1,
                   'predict_users_num': 1,
                   'predict_ads_num': 1,
                   'opt_alg': opt_alg,
                   'learning_rate': learning_rate,
                   'decay_rate': decay_rate,
                   'embedding_dim': 18,
                   'delay_compensation_para': 0,
                   'sync_parameter': 100,
                   'historical_model_num': 5                # max number of historical models kept in historical_model_dict
                   }

#======================================= Shared variables among processes ================================================
classification_queue        = multiprocessing.Manager().list()
classification_queue_lock   = multiprocessing.Lock()
get_update_queue            = multiprocessing.Manager().list()
get_update_queue_lock       = multiprocessing.Lock()
return_model_queue          = multiprocessing.Manager().list()  # received updates from these workers and will return the updated model to them
return_model_queue_lock     = multiprocessing.Lock()            # for return_model_queue
gathered_weights_dict       = multiprocessing.Manager().dict()  # key: client.ID; value: weights
batches_info_dict           = multiprocessing.Manager().dict()  # key: client.ID; value: next_batches_info
layer_name_queue            = multiprocessing.Manager().list()
historical_model_dict       = multiprocessing.Manager().dict()  # key: str(time.time()); value: model
timestamp_queue             = multiprocessing.Manager().list()  # keep all saved models' timestamps orderly, used when deleting old models
valid_updates_queue         = multiprocessing.Manager().list()
valid_updates_dict          = multiprocessing.Manager().dict()
return_model_only_queue     = multiprocessing.Manager().list()
returning_model_queue       = multiprocessing.Manager().list()  # the queue of workers who will soon receive the global model
returning_model_queue_lock  = multiprocessing.Lock()            # for returning queue
return_model_queue_lock2    = multiprocessing.Lock()            # for valid_updates_queue and return_model_only_queue
update_model_lock           = multiprocessing.Lock()
gui_event                   = multiprocessing.Event()
gui_evaluate_result         = multiprocessing.Manager().list()
gui_round_finished          = multiprocessing.Event()
#======================================= Shared variables among threads ================================================

# kill all child processes on termination
def my_termination(sig_num, addtion):
    print ('Terminating processes...')
    os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)

signal.signal(signal.SIGTERM, my_termination)

#========================================== Threads handler functions ==================================================
def connection_acceptor(classification_queue, classification_queue_lock):
    ps_fn.accept_new_connections(classification_queue, classification_queue_lock)


def connection_classifier(classification_queue, classification_queue_lock, get_update_queue, get_update_queue_lock,\
                          hyperparameters, batches_info_dict, timestamp_queue, historical_model_dict, layer_name_queue):
    """
    communicate with new connected clients in classification_queue to know whether they need initialization or to send update,
    then make initialization or passing them to get_update_queue.
    """
    ps_fn.classify_connections(classification_queue, classification_queue_lock, get_update_queue, get_update_queue_lock,\
                               hyperparameters, batches_info_dict, timestamp_queue, historical_model_dict, layer_name_queue)


def update_receiver(get_update_queue, get_update_queue_lock, return_model_queue, return_model_queue_lock, gathered_weights_dict, gui_event):
    """
    gather updates from clients in get_update_queue and get their next batches' information,
    save updates in gathered_weights_queue, save batch information in batches_info_queue,
    then passing them to return_model_queue.
    """
    ps_fn.get_update(get_update_queue, get_update_queue_lock, return_model_queue, return_model_queue_lock, gathered_weights_dict, gui_event)


def model_updater(return_model_queue, return_model_queue_lock, gathered_weights_dict, valid_updates_queue,\
                  return_model_only_queue, return_model_queue_lock2, returning_model_queue, returning_model_queue_lock,\
                  batches_info_dict, timestamp_queue, historical_model_dict, layer_name_queue, hyperparameters,\
                  gui_evaluate_result, gui_event, gui_round_finished):
    """
    check whether it's time to update the global model according to len(valid_updates_queue) >= n,
    if true: update the global model.
    """
    ps_fn.update_model(return_model_queue, return_model_queue_lock, gathered_weights_dict, valid_updates_queue,\
                       return_model_only_queue, return_model_queue_lock2, returning_model_queue, returning_model_queue_lock,\
                       batches_info_dict, timestamp_queue, historical_model_dict, layer_name_queue, hyperparameters,\
                       gui_evaluate_result, gui_event, gui_round_finished)


def model_returner(returning_model_queue, returning_model_queue_lock, batches_info_dict,\
                   timestamp_queue, historical_model_dict, layer_name_queue, hyperparameter, gui_event):
    """
    return models to clients in returning_model_queue.
    """
    ps_fn.return_model(returning_model_queue, returning_model_queue_lock, batches_info_dict,\
                       timestamp_queue, historical_model_dict, layer_name_queue, hyperparameters, gui_event)


def get_update_queue_heartbeater(get_update_queue, get_update_queue_lock):
    """
    send and receive heart beat from clients in get_update_queue to check connection,
    delete bad clients from the queue,
    ran every 99s.
    """
    interval_time = 99
    ps_fn.get_update_queue_heartbeat(get_update_queue, get_update_queue_lock, interval_time)


def return_model_queue_heartbeater(return_model_queue, return_model_queue_lock, gathered_weights_dict, valid_updates_queue,\
                                   return_model_only_queue,return_model_queue_lock2, batches_info_dict):
    """
    send and receive heart beat from clients in return_model_queue to check connection,
    delete bad clients and their information from queues,
    ran every 99s.
    """
    interval_time = 99
    ps_fn.return_model_queue_heartbeat(return_model_queue, return_model_queue_lock, gathered_weights_dict, valid_updates_queue,\
                                       return_model_only_queue, return_model_queue_lock2, batches_info_dict, interval_time)


def returning_model_queue_heartbeater(returning_model_queue, returning_model_queue_lock):
    """
    send and receive heart beat from clients in get_update_queue to check connection,
    delete bad clients from the queue,
    ran every 99s.
    """
    interval_time = 99
    ps_fn.returning_model_queue_heartbeat(returning_model_queue, returning_model_queue_lock, interval_time)


processes = []
processes.append(multiprocessing.Process(target=connection_acceptor, name='Acceptor      ',\
                 args=(classification_queue, classification_queue_lock)))

for i in range(classifier_num):
    processes.append(multiprocessing.Process(target=connection_classifier, name='Classifier_{:d}  '.format(i),\
                     args=(classification_queue, classification_queue_lock, get_update_queue, get_update_queue_lock,\
                           hyperparameters, batches_info_dict, timestamp_queue, historical_model_dict, layer_name_queue)))

for i in range(receiver_num):
    processes.append(multiprocessing.Process(target=update_receiver, name='Receiver_{:d}    '.format(i),\
                     args=(get_update_queue, get_update_queue_lock, return_model_queue, return_model_queue_lock, gathered_weights_dict, gui_event)))

processes.append(multiprocessing.Process(target=model_updater, name='Updater       ',\
    args=(return_model_queue, return_model_queue_lock, gathered_weights_dict, valid_updates_queue,\
          return_model_only_queue, return_model_queue_lock2, returning_model_queue, returning_model_queue_lock,\
          batches_info_dict, timestamp_queue, historical_model_dict, layer_name_queue, hyperparameters,\
          gui_evaluate_result, gui_event, gui_round_finished)))

for i in range(returner_num):
    processes.append(multiprocessing.Process(target=model_returner, name='Returner_{:d}    '.format(i),\
                     args=(returning_model_queue, returning_model_queue_lock, batches_info_dict,\
                           timestamp_queue, historical_model_dict, layer_name_queue, hyperparameters, gui_event)))

processes.append(multiprocessing.Process(target=get_update_queue_heartbeater, name='HeartBeater   ',\
                 args=(get_update_queue, get_update_queue_lock)))
processes.append(multiprocessing.Process(target=return_model_queue_heartbeater, name='HeartBeater   ',\
                 args=(return_model_queue, return_model_queue_lock, gathered_weights_dict, valid_updates_queue,\
                       return_model_only_queue, return_model_queue_lock2, batches_info_dict)))
processes.append(multiprocessing.Process(target=returning_model_queue_heartbeater, name='HeartBeater   ',\
                 args=(returning_model_queue, returning_model_queue_lock)))

for p in processes:
    p.start()
#========================================== Threads handler functions ==================================================

#======================================== define GUI data update worker ================================================
class DataUpdateWorker(QThread):
    data_update_value = pyqtSignal(list)
    current_round_value = pyqtSignal(int)
    progress_bar_busy_state_value = pyqtSignal(int, int)
    progress_bar_distribute_value = pyqtSignal(int)
    progress_bar_collect_value = pyqtSignal(int)
    def __init__(self):
        super(DataUpdateWorker, self).__init__()

    def run(self):
        global gui_event, gui_evaluate_result, gui_round_finished
        current_round = 0
        while(True):
            gui_event.wait()
            gui_event.clear()
            if gui_round_finished.is_set():
                gui_round_finished.clear()
                current_round += 1
                self.current_round_value.emit(current_round)
                self.progress_bar_busy_state_value.emit(1, current_round - 1)
                self.progress_bar_distribute_value.emit(0)
                self.progress_bar_collect_value.emit(0)
                continue
            if len(gui_evaluate_result) > 0:
                temp_gui_evaluate_result = deepcopy(gui_evaluate_result)
                gui_evaluate_result[:] = []
                for entry in temp_gui_evaluate_result:
                    self.data_update_value.emit([entry[0], entry[1], entry[2]])
                self.progress_bar_busy_state_value.emit(0, current_round - 1)
            else:
                self.progress_bar_distribute_value.emit(hyperparameters['sync_parameter'] - len(returning_model_queue))
                self.progress_bar_collect_value.emit(len(return_model_queue) + len(valid_updates_queue))
#======================================== define GUI data update worker ================================================

# GUI start
app = ps_gui.QApplication(sys.argv)
gui = ps_gui.MainUi()
gui.setup_ui()
gui.setup_data_update_worker(DataUpdateWorker)
gui.set_worker_num(hyperparameters['sync_parameter'])
gui.show()
print('Server started with PID = {}'.format(os.getpid()))
sys.stdout.flush()
sys.exit(app.exec_())