##############################################################################################
# Main Script for Training and Testing the AR_Transformer on Datasets of Dynamical Systems
# Author: Javier Fañanás Anaya
# Email: javierfa@unizar.es
##############################################################################################

import sys
import libs.data as data
import libs.plot as plot

from libs.config import Config
from libs.network import Network

#Check arguments:
if len(sys.argv) != 3:
    print("Use: python ar_transformer.py <dataset> <train|test>")
    exit()

#Dataset name
dataset_name = sys.argv[1]

#Mode (train or test)
mode = sys.argv[2].lower()

#Check mode
if mode == 'train':
    test = False
elif mode == 'test':
    test = True
else:
    print("Use: python ar_transformer.py <dataset> <train|test>")
    exit()

config = Config(dataset_name)

#Load and split dataset
input_train, state_train, \
input_val, state_val, \
input_test, state_test, \
scaler_input, scaler_state = data.load_dataset(config)

network = Network(config)

#Training:
if not test:
    #Create model
    network.create_model()

    #Scale data
    input_train_scaled, state_train_scaled = data.scale_split(input_train, state_train, scaler_input, scaler_state)
    input_val_scaled, state_val_scaled = data.scale_split(input_val, state_val, scaler_input, scaler_state)

    #Preprocessing
    input_train_scaled, state_train_scaled = data.preprocess_data(input_train_scaled, state_train_scaled, config,True)
    input_val_scaled, state_val_scaled = data.preprocess_data(input_val_scaled, state_val_scaled, config,True)

    #Training
    network.train_model(input_train_scaled,state_train_scaled,input_val_scaled, state_val_scaled)

#Test
else:
    #Load model from config
    network.load_model()

    #Scale data
    input_test_scaled, state_test_scaled = data.scale_split(input_test, state_test, scaler_input, scaler_state)

    #Preprocessing
    input_test_scaled, state_test_scaled = data.preprocess_data(input_test_scaled, state_test_scaled, config, True)
    input_test, state_test = data.preprocess_data(input_test, state_test, config,False)

    #Test model
    state_nn = network.test_model(input_test_scaled, state_test_scaled, scaler_state)
    
    #Print metrics and plot test series
    r2, rmse, nrmse, mae = plot.print_metrics(config, state_test, state_nn)
    plot.plot_loss(config, network.get_model_name())
    plot.plot_all_series(config, input_test, state_test, state_nn, r2, rmse, nrmse, mae)