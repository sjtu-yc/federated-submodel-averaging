# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf
import time
import math


def uint_para_compress(delta_model_para, hyperparameters, learning_rate):
    compress_k_levels = hyperparameters['compress_k_levels']
    compress_bound = hyperparameters['compress_bound']
    if compress_k_levels < 2:
        return delta_model_para
    compressed_model = []
    k_level_delta = np.linspace(-compress_bound*learning_rate, compress_bound*learning_rate, compress_k_levels)
    dist = (2.0 * compress_bound * learning_rate) / (compress_k_levels - 1)
    for delta_model_index, para in enumerate(delta_model_para):
        para_shape=list(para.shape)
        if para.ndim == 2:
            para = para.flatten()
#        print ((para+compress_bound * learning_rate)/dist)
        argmin_less = np.floor((para+compress_bound * learning_rate)/dist).astype(np.int32)
        argmin_larger = np.ceil((para+compress_bound * learning_rate)/dist).astype(np.int32)
        prop = (para-(k_level_delta[argmin_less]))/dist
        rannum = np.random.rand(len(para))
        int_array = np.where(rannum < prop, argmin_larger, argmin_less)
        if compress_k_levels <= 2**8:
            int_array = int_array.astype(np.uint8)
        elif compress_k_levels <= 2**16:
            int_array = int_array.astype(np.uint16)
        else:
            int_array = int_array.astype(np.uint32)
        int_array = int_array.reshape(para_shape)
        compressed_model.append(int_array)
    return compressed_model
    
def recover_compression(compressed_array, hyperparameters, learning_rate):
    compress_k_levels = hyperparameters['compress_k_levels']
    compress_bound = hyperparameters['compress_bound']
    if compress_k_levels < 2:
        return compressed_array
    bound = compress_bound * learning_rate
    dist = (2.0 * bound)/(compress_k_levels - 1)
    recovered_array = compressed_array * dist - bound
    return recovered_array
        
    