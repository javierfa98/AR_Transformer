#################################################################
# Functions for Preprocessing Dynamical Systems Datasets.
# Author: Javier Fañanás Anaya
# Email: javierfa@unizar.es
#################################################################

import os
import time
import pandas as pd
import numpy as np
import re

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#Natural sort of a list
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

#Load Dataset from config
def load_dataset(config):

    print("Loading ["+config.dataset_name+"] dataset ...")
    start_time = time.time()

    #Data Shape -> (batch, length, parameters)
    input_data = []
    state_data = []

    #Read dataset
    if config.shuffle:
        list_files = os.listdir(config.dataset_dir)
    else:
        #Read files sorted if we do not shuffle
        list_files = sorted(os.listdir(config.dataset_dir), key=natural_sort_key)

    for file in list_files:
        serie_dir = os.path.join(config.dataset_dir,file)
        if os.path.isfile(serie_dir):
            serie = pd.read_csv(serie_dir, sep=',', header=0)
            serie = np.array(serie.values)

            if config.length == -1: #No batches, read full serie
                input_data.append(serie[:,:config.input_size])
                state_data.append(serie[:,config.input_size:])
            else:
                #Read batches of size length
                max_length = len(serie) // config.length * config.length
                serie = serie[:max_length, :]
                for i in range(0, len(serie), config.length):
                    input_data.append(serie[i:i+config.length, :config.input_size])
                    state_data.append(serie[i:i+config.length, config.input_size:])

    input_data = np.array(input_data,dtype=object)
    state_data = np.array(state_data,dtype=object)

    #Data splits
    input_train, input_test, state_train, state_test = train_test_split(input_data, state_data, test_size=config.test_size, random_state=0, shuffle=config.shuffle)
    input_train, input_val, state_train, state_val = train_test_split(input_train, state_train, test_size=config.val_size/(1 - config.test_size), random_state=0, shuffle=True)

    #Get scalers with training split
    input_train_flat = np.concatenate(input_train, axis=0)
    state_train_flat = np.concatenate(state_train, axis=0)

    scaler_input = MinMaxScaler(feature_range=(config.scale_min, config.scale_max))
    scaler_input.fit(input_train_flat)

    scaler_state = MinMaxScaler(feature_range=(config.scale_min, config.scale_max))
    scaler_state.fit(state_train_flat)

    #Save scalers
    create_max_min_values(scaler_input, scaler_state, config.scalers_dir)

    #Load custom test:
    if config.custom_test > 0:
        input_test = []
        state_test = []
        
        list_files = os.listdir(config.custom_test_dir)

        for file in list_files:
            serie_dir = os.path.join(config.custom_test_dir,file)
            if os.path.isfile(serie_dir):
                serie = pd.read_csv(serie_dir, sep=',', header=0)
                serie = np.array(serie.values)

                if config.length == -1: #No batches, read full serie
                    input_test.append(serie[:,:config.input_size])
                    state_test.append(serie[:,config.input_size:])
                else:
                    #Read batches of size length
                    max_length = len(serie) // config.length * config.length
                    serie = serie[:max_length, :]
                    for i in range(0, len(serie), config.length):
                        input_test.append(serie[i:i+config.length, :config.input_size])
                        state_test.append(serie[i:i+config.length, config.input_size:])

        input_test = np.array(input_test,dtype=object)
        state_test = np.array(state_test,dtype=object)

    time_elapsed = time.time() - start_time
    print("Dataset ["+config.dataset_name+"] loaded in "+ str(int(time_elapsed)) + " seconds")

    return input_train, state_train, input_val, state_val, input_test, state_test, scaler_input, scaler_state

#Scale input_data and state_data with scalers
def scale_split(input_data, state_data, scaler_input, scaler_state):
    input_scaled = []
    state_scaled = []
    batches = input_data.shape[0]
    for i in range(batches):
        i_input_scaled = scaler_input.transform(input_data[i])
        i_state_scaled = scaler_state.transform(state_data[i])

        input_scaled.append(i_input_scaled)
        state_scaled.append(i_state_scaled)

    input_scaled = np.array(input_scaled,dtype=object)
    state_scaled = np.array(state_scaled,dtype=object)

    return input_scaled, state_scaled

#Preprocess data: Input of NN = (batch, length, [input + state(t-1)])
def preprocess_data(input_data, state_data, config, training):
    input_pre = []
    state_pre = []
    input_size = input_data[0].shape[1] + state_data[0].shape[1]
    if config.length == -1:
        for batch in range(input_data.shape[0]):
            length = input_data[batch].shape[0] #Length can be different at each batch
            input_batch = np.concatenate((input_data[batch][1:], state_data[batch][:length-1]),axis=1).reshape((length-1,1,input_size)).astype('float32')
            state_batch = state_data[batch][1:length].astype('float32')
            
            if config.buffer < 2:
                print("Buffer must be >= 2")
                exit()

            else:
                #Input of NN = (1, 1, Buffer*[input(t),state(t-1)]) --> (1, 1, [B, Bt-1, Bt-2 ... Bt-TB])
                input_buff = np.zeros((input_batch.shape[0] - (config.buffer-1), input_batch.shape[1], input_batch.shape[2] * config.buffer))
                for i in range(input_buff.shape[0]):
                    for j in range(config.buffer):
                        input_buff[i,0,j*input_size:(j+1)*input_size] = input_batch[i+(config.buffer-1-j),0,:].astype('float32')
                state_buff = state_batch[config.buffer-1:].astype('float32')

                input_pre.append(input_buff)
                state_pre.append(state_buff)
        
        input_pre = np.array(input_pre,dtype=object)
        state_pre = np.array(state_pre,dtype=object)
    else:
        for batch in range(input_data.shape[0]):
            length = input_data[batch].shape[0] #Length can be different at each batch
            input_batch = np.concatenate((input_data[batch][1:], state_data[batch][:length-1]),axis=1).reshape((length-1,1,input_size))
            state_batch = state_data[batch][1:length]

            if config.buffer < 2:
                print("Buffer must be >= 2")
                exit()

            else:
                #Input of NN = (1, 1, Buffer*[input(t),state(t-1)]) --> (1, 1, [B, Bt-1, Bt-2 ... Bt-TB])
                input_buff = np.zeros((input_batch.shape[0] - (config.buffer-1), input_batch.shape[1], input_batch.shape[2] * config.buffer))
                for i in range(input_buff.shape[0]):
                    for j in range(config.buffer):
                        input_buff[i,0,j*input_size:(j+1)*input_size] = input_batch[i+(config.buffer-1-j),0,:]
                state_buff = state_batch[config.buffer-1:]

                input_pre.append(input_buff)
                state_pre.append(state_buff)
        
        input_pre = np.array(input_pre,dtype=object).astype('float32')
        state_pre = np.array(state_pre,dtype=object).astype('float32')

    if training:
        #Fix shapes to --> Input of NN = (batch, length, [input + state(t-1)])
        #TO DO: Refactor code and directlty generate (batch, length, [input + state(t-1)])
        if config.length != -1:
            series, length, _, parameters = input_pre.shape
            new_shape = (series // config.batch_size, config.batch_size, length, parameters)
            input_pre = np.reshape(input_pre[:series // config.batch_size * config.batch_size], new_shape)

            series, length, parameters = state_pre.shape
            new_shape = (series // config.batch_size,config.batch_size, length, parameters)
            state_pre = np.reshape(state_pre[:series // config.batch_size * config.batch_size], new_shape)

        else: #Length -1, different for each batch
            for i in range(input_pre.shape[0]):
                input_pre[i] = np.reshape(input_pre[i], (1,input_pre[i].shape[0],input_pre[i].shape[2]))
                state_pre[i] = np.reshape(state_pre[i], (1,state_pre[i].shape[0],state_pre[i].shape[1]))
    
    return input_pre, state_pre

#Save MAX and MIN values used to scale the parameters
def create_max_min_values(scaler_input, scaler_state, file_dir):
    input_max_values = scaler_input.data_max_
    input_min_values = scaler_input.data_min_

    state_max_values = scaler_state.data_max_
    state_min_values = scaler_state.data_min_

    f = open(file_dir, "w")

    # Write input values
    for value in input_max_values:
        f.write(str(value) + " ")
    f.write("\n")

    for value in input_min_values:
        f.write(str(value) + " ")
    f.write("\n")

    # Write state values
    for value in state_max_values:
        f.write(str(value) + " ")
    f.write("\n")

    for value in state_min_values:
        f.write(str(value) + " ")
    f.write("\n")

    f.close()