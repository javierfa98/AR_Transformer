##################################################################################
# Class for Training and Testing the AR_Transformer on Datasets of Dynamical Systems
# Author: Javier Fañanás Anaya
# Email: javierfa@unizar.es
##################################################################################
import os
import tensorflow as tf
import numpy as np
import time

class Network:

    def __init__(self, config):
        self.config = config
        self.device = self.config.device

        self.output_size = self.config.state_size
    
    def create_model(self):
        ###################### ENCODER  ######################

        batch_size = self.config.batch_size
        sequence_size = self.config.buffer - 1
        input_shape = (batch_size,sequence_size, self.config.input_size + self.config.state_size)

        model_name = self.get_model_name()
        encoder_input = tf.keras.layers.Input(batch_shape=input_shape)

        i_layer = encoder_input

        #Time Feature (With low buffer length we found better results without adding time feature)
        if self.config.time:
            time_feature = tf.linspace(1.0, -1.0, sequence_size)
            time_feature = time_feature[tf.newaxis, :, tf.newaxis]
            time_feature = tf.tile(time_feature, [batch_size, 1, 1])
            i_layer = tf.concat([i_layer, time_feature], axis=-1)

            #If d_model == n_feature, we count time feature. Else will be projected
            if self.config.d_model == self.config.input_size + self.config.state_size:
                self.config.d_model += 1

        #Projection to d_model
        if self.config.d_model != self.config.input_size + self.config.state_size:
            i_layer = tf.keras.layers.Dense(self.config.d_model, activation='relu')(i_layer)
        
        #Positional encoding (With low buffer length we found better results without positional encoding)
        if self.config.positional_add or self.config.positional_concat:
            pos = tf.range(sequence_size, dtype=tf.float32)[:, tf.newaxis]
            div = tf.exp(tf.range(0, self.config.d_model, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / self.config.d_model))

            pos_enc = np.zeros((sequence_size, self.config.d_model))
            pos_enc[:, 0::2] = np.sin((sequence_size - pos - 1) * div)
            if self.config.d_model % 2 == 0:
                pos_enc[:, 1::2] = np.cos((sequence_size - pos - 1) * div)
            else:
                pos_enc[:, 1:self.config.d_model:2] = np.cos((sequence_size - pos[:, :self.config.d_model//2] - 1) * div[:self.config.d_model//2])
            pos_enc = pos_enc[np.newaxis, ...]
            pos_enc = tf.convert_to_tensor(pos_enc, dtype=tf.float32)
            if self.config.positional_add: #Add positional encoding
                i_layer = i_layer + pos_enc[:, :tf.shape(i_layer)[1], :]
            else: #Concat positional encoding
                pos_enc= tf.tile(pos_enc, [batch_size, 1, 1])
                i_layer = tf.keras.layers.Concatenate(axis=-1)([i_layer, pos_enc[:, :tf.shape(i_layer)[1], :]])
                #Duplicate d_model
                self.config.d_model *= 2
        
        #Create encoder_layers
        for i in range(self.config.encoder_layers):
            #Multihead Attention Layer (Self-Attention)
            mh_layer = tf.keras.layers.MultiHeadAttention(num_heads=self.config.num_heads, key_dim=self.config.d_model // self.config.num_heads)(i_layer,i_layer)

            #Add&Norm mh_layer + input
            if self.config.add_norm:
                add_norm = mh_layer + i_layer
                mh_layer = tf.keras.layers.LayerNormalization(epsilon=1e-6)(add_norm)
            
            #Concatenate mh_layer and input (Duplicate d_model)
            elif self.config.concatenate:
                mh_layer = tf.keras.layers.Concatenate(axis=-1)([mh_layer, i_layer])

            #FF Layer
            dense_layer = tf.keras.layers.Dense(self.config.d_model, activation='relu')(mh_layer)
            
            #Add&Norm dense + previous Add&Norm
            if self.config.add_norm:
                add_norm = dense_layer + add_norm
                dense_layer = tf.keras.layers.LayerNormalization(epsilon=1e-6)(add_norm)

            i_layer = dense_layer

        encoder_output = i_layer

        ###################### DECODER  ######################
        decoder_input = tf.keras.layers.Input(batch_shape=(batch_size, 1, self.config.input_size + self.config.state_size))

        i_layer = decoder_input
        
        #Time Feature (With low buffer length we found better results without adding time feature)
        if self.config.time:
            time_feature = tf.ones((batch_size, 1, 1), dtype=tf.float32)
            i_layer = tf.concat([i_layer, time_feature], axis=-1)
        
        #Projection to d_model
        if self.config.d_model != self.config.input_size + self.config.state_size:
            i_layer = tf.keras.layers.Dense(self.config.d_model, activation='relu')(i_layer)
        
        #Create decoder_layers
        for i in range(self.config.decoder_layers):
            #Multihead Attention Layer (Cross-Attention)
            mh_layer = tf.keras.layers.MultiHeadAttention(num_heads=self.config.num_heads, key_dim=self.config.d_model // self.config.num_heads)(i_layer,encoder_output)

            #Add&Norm mh_layer + input
            if self.config.add_norm:
                add_norm = mh_layer + i_layer
                mh_layer = tf.keras.layers.LayerNormalization(epsilon=1e-6)(add_norm)
            
            #Concatenate mh_layer and input (Duplicate d_model)
            elif self.config.concatenate:
                mh_layer = tf.keras.layers.Concatenate(axis=-1)([mh_layer, i_layer])

            #FF Layer
            dense_layer = tf.keras.layers.Dense(self.config.d_model, activation='relu')(mh_layer)
            
            #Add&Norm dense + previous Add&Norm
            if self.config.add_norm:
                add_norm = dense_layer + add_norm
                dense_layer = tf.keras.layers.LayerNormalization(epsilon=1e-6)(add_norm)

            i_layer = dense_layer
        
        decoder_output = i_layer

        ###################### OUTPUT  ######################

        decoder_output = tf.keras.layers.Flatten()(decoder_output)
        #Linear
        output_layer = tf.keras.layers.Dense(units=self.output_size)(decoder_output)

        self.model = tf.keras.Model(inputs=[encoder_input, decoder_input], outputs=output_layer, name=model_name)
        self.model.summary()
    
    #Load model from config
    def load_model(self):

        model_name = self.get_model_name()
        model_file = os.path.join(self.config.networks_dir, model_name + '.h5')
        self.model = tf.keras.models.load_model(model_file,custom_objects={self.config.loss_name: self.config.loss})
        
        self.model.summary()

    #Create and train a model with the hyperparameters from config
    def train_model(self, input_train, state_train, input_val, state_val):

        train_size = input_train.shape[0]
        val_size = input_val.shape[0]

        model_file = os.path.join(self.config.networks_dir, self.model.name + '.h5')

        #Training with Teacher Forcing (We usually train only without Teacher Forcing)
        if not self.config.only_notf:
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.lr_tf), loss=self.config.loss)
            n_epochs = self.config.epochs_tf
            v_train_loss = []
            v_validation_loss = []
            v_validation_notf_loss = []

            #Early stopping, monitoring validation_notf
            best_val_notf_loss = 10000
            epochs_without_improvement = 0
            
            print('-----Training with Teacher Forcing-----')
            with tf.device(self.device): 
                for epoch in range(n_epochs):
                    print('---------------------------------------')
                    print('Epoch: ', epoch + 1, ' of ', n_epochs)

                    #Training
                    start_time = time.time()
                    train_loss_epoch = 0
                    for batch in range(train_size):
                        train_loss_batch = self.train_tf(input_train[batch], state_train[batch])
                        train_loss_epoch += train_loss_batch
                        print('\rTraining batch: ', batch + 1, ' of ', train_size, end="")
                    train_loss_epoch /= train_size
                    v_train_loss.append(train_loss_epoch)
                    print()

                    #Validation
                    val_loss_epoch = 0
                    for batch in range(val_size):
                        val_loss_batch = self.validation_tf(input_val[batch], state_val[batch])
                        val_loss_epoch += val_loss_batch
                        print('\rValidation batch (TF): ', batch + 1, ' of ', val_size, end="")
                    val_loss_epoch /= val_size
                    v_validation_loss.append(val_loss_epoch)
                    print()

                    #Validation (no-TF)
                    val_loss_notf_epoch = 0
                    for batch in range(val_size):
                        val_loss_batch = self.validation_notf(input_val[batch], state_val[batch])
                        val_loss_notf_epoch += val_loss_batch
                        print('\rValidation batch (no-TF): ', batch + 1, ' of ', val_size, end="")
                    val_loss_notf_epoch /= val_size
                    v_validation_notf_loss.append(val_loss_notf_epoch)

                    #Epoch stats
                    time_elapsed = time.time() - start_time
                    print('\nTime elapsed: ', round(time_elapsed,2))
                    print('Training (TF) loss = ', train_loss_epoch)
                    print('Validation (TF) loss = ', val_loss_epoch)
                    print('Validation (no-TF) loss = ',val_loss_notf_epoch)

                    #Save model if val_loss_notf has improved:
                    if val_loss_notf_epoch < best_val_notf_loss:
                        best_val_notf_loss = val_loss_notf_epoch
                        self.model.save(model_file)
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1
                        print('Epochs without Validation (no-TF) improvement: ', epochs_without_improvement)
                        if epochs_without_improvement == self.config.early_stop_tf:
                            print('Early stopped')
                            break  
                print('---------------------------------------')

                #Save loss data:
                np_loss = np.array([v_train_loss,v_validation_loss, v_validation_notf_loss])
                loss_path = os.path.join(self.config.losses_dir, self.model.name + '_tf.txt')
                np.savetxt(loss_path, np_loss) 

        #Training Without Teacher Forcing
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.lr_notf[0]),loss=self.config.loss)

        lr_i = 0
        lr_max = len(self.config.lr_notf)
        n_epochs = self.config.epochs_notf

        v_train_loss = []
        v_validation_notf_loss = []

        #Early stopping, monitoring validation_notf
        best_val_notf_loss = 10000
        epochs_without_improvement = 0

        print('-----Training without Teacher Forcing-----')
        with tf.device(self.device):
            for epoch in range(n_epochs):
                print('---------------------------------------')
                print('Epoch: ', epoch + 1, ' of ', n_epochs)

                #Training (no-TF)
                start_time = time.time()
                train_loss_epoch = 0
                for batch in range(train_size):
                    train_loss_batch = self.train_notf(input_train[batch], state_train[batch])
                    train_loss_epoch += train_loss_batch
                    print('\rTraining batch: ', batch + 1, ' of ', train_size, end="")
                train_loss_epoch /= train_size
                v_train_loss.append(train_loss_epoch)
                print()

                #Validation (no-TF)
                val_loss_notf_epoch = 0
                for batch in range(val_size):
                    val_loss_batch = self.validation_notf(input_val[batch], state_val[batch])
                    val_loss_notf_epoch += val_loss_batch
                    print('\rValidation batch (no-TF): ', batch + 1, ' of ', val_size, end="")
                val_loss_notf_epoch /= val_size
                v_validation_notf_loss.append(val_loss_notf_epoch)

                #Epoch stats
                time_elapsed = time.time() - start_time
                print('\nTime elapsed: ', round(time_elapsed,2))
                print('Training (no-TF) loss = ', train_loss_epoch)
                print('Validation (no-TF) loss = ',val_loss_notf_epoch)

                #Save model if val_loss_notf has improved:
                if val_loss_notf_epoch < best_val_notf_loss:
                    best_val_notf_loss = val_loss_notf_epoch
                    self.model.save(model_file)
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    print('Epochs without Validation (no-TF) improvement: ', epochs_without_improvement)
                    if epochs_without_improvement == self.config.early_stop_notf:
                        #Reduce LR
                        lr_i += 1
                        if lr_i < lr_max:
                            print('Reducing learning rate to: '+ str(self.config.lr_notf[lr_i])) 
                            epochs_without_improvement = 0
                            self.model = tf.keras.models.load_model(model_file,custom_objects={self.config.loss_name: self.config.loss})
                            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.lr_notf[lr_i]), loss=self.config.loss)
                        else:
                            print('Minimun learning rate reached. Early stopped.')
                            break
        
            print('---------------------------------------')

            #Save loss data:
            np_loss = np.array([v_train_loss, v_validation_notf_loss])
            loss_path = os.path.join(self.config.losses_dir, self.model.name + '_notf.txt')
            np.savetxt(loss_path, np_loss)
    
    def test_model(self, input_test, state_test, scaler_state):
        start_time = time.time()
        n_series = input_test.shape[0]

        #Iterate over all test series:
        #Test (no-TF)
        print('---------- Testing ', n_series,' batches  ----------')
        print('Model: ',self.get_model_name())
        test_loss_notf = 0

        state_nn = [] #Save state predictions at original scale
        with tf.device(self.device):
            for serie in range(n_series):
                #Test loss
                test_loss_notf_batch, state_nn_batch = self.test_notf(input_test[serie], state_test[serie], scaler_state)
                test_loss_notf += test_loss_notf_batch
                print('\rTest batch (no-TF): ', serie + 1, ' of ', n_series, end="")

                for state in state_nn_batch:
                    state_nn.append(state)

            
            test_loss_notf /= n_series
            time_elapsed = time.time() - start_time
            print('\nTime elapsed: ', round(time_elapsed,2))
            print('\nTest loss (no-TF) = ',test_loss_notf)
            print('---------------------------------------')

        return state_nn
    
    def generate_input(self, input_t):
        input_reshape = input_t.reshape((input_t.shape[0],self.config.buffer, self.config.input_size + self.config.state_size))
        return [input_reshape[:,1:, :], input_reshape[:,0:1,:]]
    
    #Trainining with TF
    def train_tf(self, input_train, state_train):
        length = input_train.shape[1]
        state_nn_batch = []
        with tf.GradientTape() as tape:
            for t in range(length):
                input_t = input_train[:,t:t+1,:]
                #Reshape for multihead attention shape
                state_nn = self.model(self.generate_input(input_t),training=True)
                state_nn_batch.append(state_nn)
            state_nn_batch = tf.stack(state_nn_batch,axis=1)
            train_loss_batch = self.model.loss(state_train, state_nn_batch)    
        gradients = tape.gradient(train_loss_batch, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        #self.model.reset_states() #For memory cells
        return tf.reduce_mean(train_loss_batch).numpy().item()
    
    #Validation with TF
    def validation_tf(self, input_val, state_val):
        length = input_val.shape[1]
        state_nn_batch= []
        for t in range(length):
            input_t = input_val[:,t:t+1,:]
            #Reshape for multihead attention shape
            state_nn = self.model(self.generate_input(input_t))
            state_nn_batch.append(state_nn)
        state_nn_batch = tf.stack(state_nn_batch,axis=1)
        val_loss_batch = self.model.loss(state_val, state_nn_batch)
        #self.model.reset_states() #For memory cells
        return tf.reduce_mean(val_loss_batch).numpy().item()
    
    #Training with no-TF
    def train_notf(self, input_train, state_train):
        length = input_train.shape[1]
        state_nn_batch= []
        with tf.GradientTape() as tape:
            #We know the state at t=0
            input_t = input_train[:,0:1,:]
            #Reshape for multihead attention shape
            state_nn = self.model(self.generate_input(input_t),training=True)
            state_nn_batch.append(state_nn)
            for t in range(1,length):
                #Modify actual input with next known input and previous state prediction
                input_t = np.roll(input_t, self.config.input_size+self.config.state_size, axis=-1)
                input_t[:,0,:self.config.input_size] = input_train[:,t,:self.config.input_size]
                input_t[:,0,self.config.input_size:self.config.input_size+self.config.state_size] = state_nn

                #Get next state
                #Reshape for multihead attention shape
                state_nn = self.model(self.generate_input(input_t),training=True)
                state_nn_batch.append(state_nn)
            state_nn_batch = tf.stack(state_nn_batch,axis=1)
            train_loss_batch = self.model.loss(state_train, state_nn_batch)
        gradients = tape.gradient(train_loss_batch, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        #self.model.reset_states() #For memory cells
        return tf.reduce_mean(train_loss_batch).numpy().item()
    
    #Validation with no-TF
    def validation_notf(self, input_val, state_val):
        length = input_val.shape[1]
        state_nn_batch= []
        input_t = input_val[:,0:1,:]
        #Reshape for multihead attention shape
        state_nn = self.model(self.generate_input(input_t))
        state_nn_batch.append(state_nn)
        for t in range(1,length):
            #Modify actual input with next known input and previous state prediction
            input_t = np.roll(input_t, self.config.input_size+self.config.state_size, axis=-1)
            input_t[:,0,:self.config.input_size] = input_val[:,t,:self.config.input_size]
            input_t[:,0,self.config.input_size:self.config.input_size+self.config.state_size] = state_nn
            #Get next state
            #Reshape for multihead attention shape
            state_nn = self.model(self.generate_input(input_t))
            state_nn_batch.append(state_nn)
        state_nn_batch = tf.stack(state_nn_batch,axis=1)
        val_loss_batch = self.model.loss(state_val, state_nn_batch)
        #self.model.reset_states() #For memory cells
        return tf.reduce_mean(val_loss_batch).numpy().item()
    
    #Testing with no-TF
    def test_notf(self, input_test, state_test, scaler_state):
        length = input_test.shape[1]
        state_nn_batch = []
        v_state_nn = np.zeros((self.config.batch_size,length, self.output_size))
        input_t = input_test[:,0:1,:]
        #Reshape for multihead attention shape
        state_nn = self.model(self.generate_input(input_t))
        state_nn_batch.append(state_nn)
        v_state_nn[:,0,:] = scaler_state.inverse_transform(state_nn)
        for t in range(1,length):
            #Modify actual input with next known input and previous state prediction
            input_t = np.roll(input_t, self.config.input_size+self.config.state_size, axis=-1)
            input_t[:,0,:self.config.input_size] = input_test[:,t,:self.config.input_size]
            input_t[:,0,self.config.input_size:self.config.input_size+self.config.state_size] = state_nn
            #Get next state
            #Reshape for multihead attention shape
            state_nn = self.model(self.generate_input(input_t))
            state_nn_batch.append(state_nn)
            v_state_nn[:,t,:] = scaler_state.inverse_transform(state_nn)
        state_nn_batch = tf.stack(state_nn_batch,axis=1)
        test_loss_batch = self.model.loss(state_test, state_nn_batch)
        #self.model.reset_states() #For memory cells
        return tf.reduce_mean(test_loss_batch).numpy().item(), v_state_nn
    
    def get_model_name(self):
        model_name = self.config.dataset_name

        model_name = model_name + "_tdl_" + str(self.config.buffer)
 
        model_name = model_name + "_encoder_"+str(self.config.encoder_layers)+ "_decoder_"+str(self.config.decoder_layers)
        model_name = model_name +"_h_"+str(self.config.num_heads) +"_dmodel_"+str(self.config.d_model)
        
        if self.config.time:
            model_name = model_name +"_time"

        if self.config.positional_add:
            model_name = model_name +"_positional_add"

        if self.config.positional_concat:
            model_name = model_name +"_positional_concat"

        if self.config.add_norm:
            model_name = model_name +"_addnorm"

        if self.config.concatenate:
            model_name = model_name +"_concat"

        if not self.config.only_notf:
            model_name = model_name + "_epochs_"+str(self.config.epochs_tf)+"_" +str(self.config.epochs_notf)
        else:
            model_name = model_name + "_epochs_" +str(self.config.epochs_notf)
            
        model_name = model_name +"_loss_"+self.config.loss_name

        if self.config.length == -1:
            model_name = model_name + "_length_full"
        else:
            model_name = model_name + "_length_" + str(self.config.length)

        model_name = model_name + "_batch_" + str(self.config.batch_size)

        return model_name