##################################################################################
# Configuration of all the Datasets and Networks
# Author: Javier Fañanás Anaya
# Email: javierfa@unizar.es
##################################################################################
import os

import libs.metrics as metrics

class Config:
    def __init__(self, dataset_name):

        self.dataset_name = dataset_name
        self.dataset_dir = os.path.join('datasets',dataset_name,'series') 
        self.scalers_dir = os.path.join('datasets',dataset_name,'scalers')
        self.networks_dir = os.path.join('datasets',dataset_name,'networks')
        self.losses_dir = os.path.join('datasets',dataset_name,'losses')
        
        #If there is no GPU available, use '/cpu:0'
        self.device = '/gpu:0'

        # Industrial Robot Dataset: 
        # https://www.nonlinearbenchmark.org/benchmarks/industrial-robot
        if dataset_name == 'industrial_robot':
            
            #DATA CONFIG:
            #Validation and Test size
            self.val_size = 0.072
            self.test_size = 0.082
            #Shuffle all the series
            self.shuffle = True 
            self.custom_test = 0

            #Batch lenght (Prediction horizon during training/validation)
            #With length = -1 AND batch_size == 1, each data series will have its maximum length
            self.length =  600 
            self.batch_size = 6 

            #Buffer/Historical
            #Encoder length = buffer-1
            self.buffer = 30 # >= 2

            #Input: u (Nm)
            self.input_size = 6 
            self.input_labels = ["u_1","u_2","u_3","u_4","u_5","u_6"]
            #State: y (deg)
            self.state_size = 6
            self.state_labels = ["y_1","y_2","y_3","y_4","y_5","y_6"]

            #Data scale
            self.scale_min = -1
            self.scale_max = 1

            #NETWORK CONFIG:
            self.encoder_layers = 3
            self.decoder_layers = 1
            self.num_heads = 8
            #self.d_model = self.input_size + self.state_size
            self.d_model = 128

            #Add a time feature:
            self.time = False
            #Positional encoding:
            self.positional_add = False #Add Positional Encoding (Default)
            self.positional_concat = False #Concat Positional Encoding

            #AddNorm layers or concatenate after Attention and Dense layer.
            self.add_norm = False #Residual connection and Normalization after Attention and Dense Layer
            self.concatenate = True #Concatenate data after Attention

            #TRAINING CONFIG:
            #optimizer = Adam
            self.loss = metrics.r2
            self.loss_name = 'r2'
            self.only_notf = True #Training ONLY without Teacher-Forcing (TF)

            self.epochs_tf = 50 #Max epochs during TF training
            self.lr_tf = 0.001 #Learning Rate used during TF training
            self.early_stop_tf = 3 #Early stop during TF Training

            self.epochs_notf = 200 #Max epochs during no-TF training
            self.lr_notf  = [0.001, 0.0001, 0.00001] #Learning rate used during no-TF training
            self.early_stop_notf = 5 #Early stop during no-TF Training

        # 3 DOF Robot Arm Dataset 
        # https://github.com/rr-learning/transferable_dynamics_dataset
        elif dataset_name == 'robot_arm':

            #DATA CONFIG:
            #Validation and Test size
            self.val_size = 0.06
            self.test_size = 0.18
            #Shuffle all the series
            self.shuffle = False
            self.custom_test = 0 #0 --> test_size from /datasets/robot_arm/series
                                 #1, 2, 3 --> Tests from /datasets/robot_arm/series/test_

            #Batch lenght (Prediction horizon during training/validation)
            #With length = -1 AND batch_size == 1, each data series will have its maximum length
            self.length =  1000
            self.batch_size = 7 

            #Buffer/Historical
            #Encoder length = buffer-1
            self.buffer = 30 # >= 2

            #Input: u_torque_1, u_torque_2, u_torque_3 (Nm)
            self.input_size = 3
            self.input_labels = ["u_torque_1", "u_torque_2", "u_torque_3"]
            #State: y_angle_1, y_angle_2, y_angle_3 (rad), y_vel_1, y_vel_2, y_vel_3 (rad/s)
            self.state_size = 6
            self.state_labels = ["y_angle_1", "y_angle_2", "y_angle_3", "y_vel_1", "y_vel_2", "y_vel_3"]

            #Data scale
            self.scale_min = -1
            self.scale_max = 1

            #NETWORK CONFIG:
            self.encoder_layers = 3
            self.decoder_layers = 1
            self.num_heads = 8
            #self.d_model = self.input_size + self.state_size
            self.d_model = 128

            #Add a time feature:
            self.time = False
            #Positional encoding:
            self.positional_add = False #Add Positional Encoding (Default)
            self.positional_concat = False #Concat Positional Encoding

            #AddNorm layers or concatenate after Attention and Dense layer.
            self.add_norm = False #Residual connection and Normalization after Attention and Dense Layer
            self.concatenate = True #Concatenate data after Attention

            #TRAINING CONFIG:
            #optimizer = Adam
            self.loss = metrics.mse
            self.loss_name = 'mse'
            self.only_notf = True #Training ONLY without Teacher-Forcing (TF)

            self.epochs_tf = 50 #Max epochs during TF training
            self.lr_tf = 0.001 #Learning rate used during TF training
            self.early_stop_tf = 3 #Early stop during TF Training

            self.epochs_notf = 200 #Max epochs during no-TF training
            self.lr_notf  = [0.001, 0.0001, 0.00005] #Learning rate used during no-TF training
            self.early_stop_notf = 5 #Early stop during no-TF Training
        else:
            print("Dataset ["+ dataset_name +"] not found")
            exit()
        
        self.custom_test_dir = os.path.join(self.dataset_dir,"test_"+str(self.custom_test)) 