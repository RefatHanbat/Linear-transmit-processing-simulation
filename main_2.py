import numpy as np

import copy

from fParam import*

from fCalculations import * 

from fChannel import *

from fAlgorithm import*

from fPlot import*

from tqdm import tqdm

sys_param = myf_sys_param()

Num_samples = 10000

x_axis_name = "EsoverNO_dB_cand"

if(x_axis_name == "EsoverNO_dB_cand"):

    x_axis_cand = np.arange(start=-10, stop=35, step=5)

Num_algorithms = 6

MSE = np.zeros((Num_algorithms,np.size(x_axis_cand,axis=0)))

for ind1 in tqdm(range(0,Num_samples)):

    # Step (1): Generate random channel

    param_channel = {}

    param_channel["seed_seq"] = Num_samples + ind1

    channel = myf_channel(sys_param, param_channel)

    MSE_temp = np.zeros((Num_algorithms, np.size(x_axis_cand, axis=0)))

    for ind2 in range(0, np.size(x_axis_cand, axis=0)):

        if (x_axis_name == "EsoverNO_dB_cand"):

            sys_param["EsoverNO_dB"] = x_axis_cand[ind2] # Es/N0 [dB]

            sys_param["EsoverNO"] = 10**(sys_param["EsoverNO_dB"]/10) # [no unit]

            sys_param["NO"] = sys_param["Es"]/sys_param["EsoverNO"] # Noise energy [Joule]


        P_mat, G_mat, Ns_opt = myf_P(sys_param)

        J_rx = myf_J_rx(sys_param,channel,P_mat)

        ### Algorithm 1 ########

        solution_algorithm_1 = myf_algorithm_3(sys_param,J_rx,mode="RxMF")

        MSE_temp[0][ind2] = np.abs(solution_algorithm_1["epsilon_mat"])

        #### Algoritm 2 #######

        solution_algorithm_2 = myf_algorithm_3(sys_param,J_rx,mode="RxZF")

        MSE_temp[1][ind2] = np.abs(solution_algorithm_2["epsilon_mat"])

        ###### Algorithm 3 ######

        solution_algorithm_3 = myf_algorithm_3(sys_param,J_rx,mode="RxWF")

        MSE_temp[2][ind2] = np.abs(solution_algorithm_3["epsilon_mat"])

        ### Algorithm 4 ######

        J_tx = myf_J_tx(sys_param,channel,G_mat)

        solution_algorithm_4 = myf_algorithm_4(sys_param,J_tx,mode="TxMF")

        MSE_temp[3][ind2] = np.abs(solution_algorithm_4["epsilon_mat"])
        

        #### Algorihtm 5 ####

        solution_algorithm_5 = myf_algorithm_4(sys_param,J_tx,mode="TxZF")

        MSE_temp[4][ind2] = np.abs(solution_algorithm_5["epsilon_mat"])

        #### Algorithm 6 ####

        solution_algorithm_5 = myf_algorithm_4(sys_param,J_tx,mode="TxWF")

        MSE_temp[5][ind2] = np.abs(solution_algorithm_5["epsilon_mat"])
        
###Accumulation

MSE = MSE + MSE_temp

myf_plot_MSE(sys_param,x_axis_cand,MSE,x_axis_name)