##################################################################################
# This file defines functions to plot the loss evolution during training with
# and without teacher forcing, calculate and print various performance metrics, 
# and visualize the prediction results for different datasets.
##################################################################################
# Author: Javier Fañanás Anaya
# Email: javierfa@unizar.es
##################################################################################

import libs.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt
import os

#Colors
COLOR_1 = "#130303"
COLOR_2 = "#ff715b"
COLOR_3 = "#53A2B2"
COLOR_4 = "#FDCA40"

#Plot loss evolution with TF and with no-TF
def plot_loss(config, model_name):
    #With TF
    model_path_txt = os.path.join(config.losses_dir, model_name +"_tf.txt")
    if os.path.isfile(model_path_txt):
        np_loss = np.loadtxt(model_path_txt)

        training_loss = np_loss[0]
        validation_loss = np_loss[1]
        validation_notf_loss = np_loss[2]

        plt.figure()
        plt.plot(training_loss, label="Training (TF)", color = COLOR_1)
        plt.plot(validation_loss, label="Validation (TF)", color = COLOR_2)
        plt.plot(validation_notf_loss, label="Validation (no-TF)", color = COLOR_3)
        plt.legend(loc='best')
        plt.title('Losses (Training with teacher forcing)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
    #Without TF
    model_path_txt = os.path.join(config.losses_dir, model_name +"_notf.txt")
    if os.path.isfile(model_path_txt):
        np_loss = np.loadtxt(model_path_txt)

        training_loss = np_loss[0]
        validation_loss = np_loss[1]

        plt.figure()
        plt.plot(training_loss, label="Training (no-TF)", color = COLOR_1)
        plt.plot(validation_loss, label="Validation (no-TF)", color = COLOR_3)
        plt.legend(loc='best')
        plt.title('Losses (Training without teacher forcing)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

def print_metrics(config, state_test, state_nn):
    n_series = len(state_nn)

    nrmse = np.zeros((n_series, config.state_size))
    rmse = np.zeros((n_series, config.state_size))
    r2 = np.zeros((n_series, config.state_size))
    mae = np.zeros((n_series, config.state_size))

    for serie in range(n_series):
        for parameter in range(config.state_size):
            nrmse[serie, parameter] = metrics.nrmse_2dim(state_test[serie][:,parameter].astype(np.float32), state_nn[serie][:,parameter].astype(np.float32))
            rmse[serie, parameter] = metrics.rmse_2dim(state_test[serie][:,parameter].astype(np.float32), state_nn[serie][:,parameter].astype(np.float32))
            r2[serie, parameter] = - metrics.r2_2dim(state_test[serie][:,parameter].astype(np.float32), state_nn[serie][:,parameter].astype(np.float32))
            mae[serie, parameter] = metrics.mae_2dim(state_test[serie][:,parameter].astype(np.float32), state_nn[serie][:,parameter].astype(np.float32))

    print("NRMSE:")
    for i in range(config.state_size):
        if nrmse.shape[0] > 1:
            metric_str = str(round(np.mean(nrmse[:, i]), 4)) + ' \u00B1 ' + str(round(np.std(nrmse[:, i]), 4))
        else:
            metric_str = str(round(np.mean(nrmse[:, i]), 4))
        metric_str = config.state_labels[i] + " = " + metric_str
        print(metric_str)
    all_str = "All = " + str(round(np.mean(nrmse), 4)) + ' \u00B1 ' + str(round(np.std(nrmse), 4))
    print(all_str)
    print("\nRMSE:")
    for i in range(config.state_size):
        if nrmse.shape[0] > 1:
            metric_str = str(round(np.mean(rmse[:, i]), 4)) + ' \u00B1 ' + str(round(np.std(rmse[:, i]), 4))
        else:
            metric_str = str(round(np.mean(rmse[:, i]), 4))
        metric_str = config.state_labels[i] + " = " + metric_str
        print(metric_str)
    print("\nMAE:")
    for i in range(config.state_size):
        if mae.shape[0] > 1:
            metric_str = str(round(np.mean(mae[:, i]), 4)) + ' \u00B1 ' + str(round(np.std(mae[:, i]), 4))
        else:
            metric_str = str(round(np.mean(mae[:, i]), 4))
        metric_str = config.state_labels[i] + " = " + metric_str
        print(metric_str)
    print('\nR\u00b2:')
    for i in range(config.state_size):
        if nrmse.shape[0] > 1:
            metric_str = str(round(np.mean(r2[:, i]), 4)) + ' \u00B1 ' + str(round(np.std(r2[:, i]), 4))
        else:
            metric_str = str(round(np.mean(r2[:, i]), 4))
        metric_str = config.state_labels[i] + " = " + metric_str
        print(metric_str)
    all_str = "All = " + str(round(np.mean(r2), 4)) + ' \u00B1 ' + str(round(np.std(r2), 4))
    print(all_str)
    print('---------------------------------------')

    return r2, rmse, nrmse, mae

def print_metrics_i(config, r2, rmse, nrmse, mae):
    print("NRMSE:")
    for i in range(config.state_size):
        metric_str = str(round(nrmse[i], 4))
        metric_str = config.state_labels[i] + " = " + metric_str
        print(metric_str)
    all_str = "All = " + str(round(np.mean(nrmse), 4)) + ' \u00B1 ' + str(round(np.std(nrmse), 4))
    print(all_str)
    print("\nRMSE:")
    for i in range(config.state_size):
        metric_str = str(round(rmse[i], 4))
        metric_str = config.state_labels[i] + " = " + metric_str
        print(metric_str)
    print("\nMAE:")
    for i in range(config.state_size):
        metric_str = str(round(mae[i], 4))
        metric_str = config.state_labels[i] + " = " + metric_str
        print(metric_str)
    print('\nR\u00b2:')
    for i in range(config.state_size):
        metric_str = str(round(r2[i], 4))
        metric_str = config.state_labels[i] + " = " + metric_str
        print(metric_str)
    all_str = "All = " + str(round(np.mean(r2), 4)) + ' \u00B1 ' + str(round(np.std(r2), 4))
    print(all_str)
    print('---------------------------------------')

def plot_all_series(config, input_test, state_test, state_nn, r2, rmse, nrmse, mae):
    n_series = len(state_nn)
    for serie in range(n_series):
        print("Test serie = ",serie)
        print_metrics_i(config, r2[serie], rmse[serie], nrmse[serie], mae[serie])
        plot_serie_i(config, input_test[serie], state_test[serie], state_nn[serie])

#Plot the estimation of the model with the actual values
def plot_serie_i(config, input_test, state_test, state_nn):
    length = range(len(state_nn))

    # Industrial Robot Dataset: 
    # https://www.nonlinearbenchmark.org/benchmarks/industrial-robot
    if config.dataset_name == 'industrial_robot':
        plt.figure(1, figsize=(6, 10))
        
        font_size = '8'

        plt.subplot(6, 1, 1)
        plt.plot(length, state_test[:, 0], color=COLOR_2, label="$q_1$")
        plt.plot(length, state_nn[:, 0], color=COLOR_1, linestyle='--', label="Prediction")
        ax = plt.gca()
        ax.set_ylim([-125, 125])
        plt.legend(loc='upper left', fontsize=font_size)
        plt.ylabel('$q_1$ (deg)')
        plt.title('Industrial Robot Benchmark')

        plt.subplot(6, 1, 2)
        plt.plot(length, state_test[:, 1], color=COLOR_2, label="$q_2$")
        plt.plot(length, state_nn[:, 1], color=COLOR_1, linestyle='--', label="Prediction")
        ax = plt.gca()
        ax.set_ylim([-125, 125])
        plt.legend(loc='upper left', fontsize=font_size)
        plt.ylabel('$q_2$ (deg)')

        plt.subplot(6, 1, 3)
        plt.plot(length, state_test[:, 2], color=COLOR_2, label="$q_3$")
        plt.plot(length, state_nn[:, 2], color=COLOR_1, linestyle='--', label="Prediction")
        ax = plt.gca()
        ax.set_ylim([-125, 125])
        plt.legend(loc='upper left', fontsize=font_size)
        plt.ylabel('$q_3$ (deg)')

        plt.subplot(6, 1, 4)
        plt.plot(length, state_test[:, 3], color=COLOR_2, label="$q_4$")
        plt.plot(length, state_nn[:, 3], color=COLOR_1, linestyle='--', label="Prediction")
        ax = plt.gca()
        ax.set_ylim([-125, 125])
        plt.legend(loc='upper left', fontsize=font_size)
        plt.ylabel('$q_4$ (deg)')

        plt.subplot(6, 1, 5)
        plt.plot(length, state_test[:, 4], color=COLOR_2, label="$q_5$")
        plt.plot(length, state_nn[:, 4], color=COLOR_1, linestyle='--', label="Prediction")
        ax = plt.gca()
        ax.set_ylim([-125, 125])
        plt.legend(loc='upper left', fontsize=font_size)
        plt.ylabel('$q_5$ (deg)')

        plt.subplot(6, 1, 6)
        plt.plot(length, state_test[:, 5], color=COLOR_2, label="$q_6$")
        plt.plot(length, state_nn[:, 5], color=COLOR_1, linestyle='--', label="Prediction")
        ax = plt.gca()
        ax.set_ylim([-125, 125])
        plt.legend(loc='upper left', fontsize=font_size)
        plt.xlabel('Time step (0.1s)')
        plt.ylabel('$q_6$ (deg)')

        plt.subplots_adjust(left=0.13, bottom=0.05, right=0.99, top=0.97, hspace=0.30)
        plt.show()
    
    # 3 DOF Robot Arm Dataset 
    # https://github.com/rr-learning/transferable_dynamics_dataset
    elif config.dataset_name == 'robot_arm':
        plt.figure(1, figsize=(12, 5))
        
        font_size = '8'

        plt.subplot(3, 2, 1)
        plt.plot(length, state_test[:, 0], color=COLOR_2, label=r"$\theta_1$")
        plt.plot(length, state_nn[:, 0], color=COLOR_1, linestyle='--', label=r"$\hat{\theta}_1$")
        ax = plt.gca()
        ax.set_ylim([0, 3])
        plt.legend(loc='best', fontsize=font_size)
        plt.ylabel('Angle (rad)')

        plt.subplot(3, 2, 3)
        plt.plot(length, state_test[:, 1], color=COLOR_2, label=r"$\theta_2$")
        plt.plot(length, state_nn[:, 1], color=COLOR_1, linestyle='--', label=r"$\hat{\theta}_2$")
        ax = plt.gca()
        ax.set_ylim([0, 3])
        plt.legend(loc='best', fontsize=font_size)
        plt.ylabel('Angle (rad)')

        plt.subplot(3, 2, 5)
        plt.plot(length, state_test[:, 2], color=COLOR_2, label=r"$\theta_3$")
        plt.plot(length, state_nn[:, 2], color=COLOR_1, linestyle='--', label=r"$\hat{\theta}_3$")
        ax = plt.gca()
        ax.set_ylim([0, 3])
        plt.legend(loc='best', fontsize=font_size)
        plt.ylabel('Angle (rad)')
        plt.xlabel('Steps')

        plt.subplot(3, 2, 2)
        plt.plot(length, state_test[:, 3], color=COLOR_2, label=r"$\dot{\theta}_1$")
        plt.plot(length, state_nn[:, 3], color=COLOR_1, linestyle='--', label=r"$\hat{\dot{\theta}}_1$")
        ax = plt.gca()
        ax.set_ylim([-20, 20])
        plt.legend(loc='best', fontsize=font_size)
        plt.ylabel('Angular Vel. (rad/s)')

        plt.subplot(3, 2, 4)
        plt.plot(length, state_test[:, 4], color=COLOR_2, label=r"$\dot{\theta}_2$")
        plt.plot(length, state_nn[:, 4], color=COLOR_1, linestyle='--', label=r"$\hat{\dot{\theta}}_2$")
        ax = plt.gca()
        ax.set_ylim([-20, 20])
        plt.legend(loc='best', fontsize=font_size)
        plt.ylabel('Angular Vel. (rad/s)')

        plt.subplot(3, 2, 6)
        plt.plot(length, state_test[:, 5], color=COLOR_2, label=r"$\dot{\theta}_3$")
        plt.plot(length, state_nn[:, 5], color=COLOR_1, linestyle='--', label=r"$\hat{\dot{\theta}}_3$")
        ax = plt.gca()
        ax.set_ylim([-20, 20])
        plt.legend(loc='best', fontsize=font_size)
        plt.ylabel('Angular Vel. (rad/s)')
        plt.xlabel('Steps')


        plt.subplots_adjust(left=0.04, bottom=0.09, right=0.99, top=0.97, hspace=0.30)
        plt.show()