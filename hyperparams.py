#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 11:10:44 2020

@author: adam
"""
import os
import numpy as np

class HyperParams(object):

    
    # roberta-base
    #bert_model_path = 'roberta-base'
    bert_model_path = 'bert-base-uncased'
    #bert_model_path = './albert'

    data_name = "cmu_mosei"
    real_path = os.getcwd()

    if data_name == "cmu_mosi":
        #available_emotions = ["-3", "-2", "-1", "0", '1', '2', '3']
        #categorical_map =  {'-3': 0, '-2': 1, '-1': 2, '0': 3, '1': 4, '2': 5, '3': 6} 
        available_emotions = ["-1", "1"]
        categorical_map =  {'-3': 0, '-2': 0 , '2': 1, '3': 1}
        self_dataset = real_path + "/datasets/cmu_mosi/cmu_mosi.pkl"
        n_classes = 2 #4

    if data_name == "cmu_mosei":
        # available_emotions = ["-3", "-2", "-1", "0", '1', '2', '3']
        # categorical_map =  {'-3': 0, '-2': 1, '-1': 2, '0': 3, '1': 4, '2': 5, '3': 6}
        available_emotions = ["-1", "1"]
        categorical_map =  {'-3': 0, '-2': 0 , '2': 1, '3': 1}
        self_dataset = real_path + "/datasets/cmu_mosei/"
        n_classes = 2 #4

    if data_name == "iemocap":
        #available_emotions = ['neu','hap','sad','ang']
        #categorical_map =  {'neu':0,'hap':1,'sad':2,'ang':3}
        available_emotions = ['hap','sad']
        categorical_map = {'hap':0,'sad':1}
        self_dataset = real_path + "/datasets/iemocap/iemocap.pkl"
        n_classes = 2 #4
   
    # 语音数据的初始维度
    speech_h = 300
    speech_w = 200

    # maxlen speech length
    max_speech_length = 100
    max_sequence_length = 100
    
    # 测试集所占百分比
    test_size = 0.2
    batch_size = 16
    vaild_batch_size = 16
    epochs = 50


    # 维度，与bert保持一致
    embed_dim = 768
    hidden_size = 768

    '''-------------------参-------------------'''
    num_heads = 4
    patience = 3
    weight_decay = 1e-4
    decay_rate = 0.8
    grad_clip = 2.
    learn_rate = 3e-5
    min_learn_rate = 1e-5
    dropout_rate = 0.5
    '''-------------------参-------------------'''
    print("data_name : ", data_name)
    print(f"num_heads : {num_heads}")
    print(f"patience : {patience}")
    print(f"weight_decay : {weight_decay}")
    print(f"grad_clip : {grad_clip}")
    print(f"learn_rate : {learn_rate}")
    print(f"min_learn_rate : {min_learn_rate}")
    print(f"dropout_rate : {dropout_rate}")
