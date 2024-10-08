import numpy as np

import copy

from fParam import*

from fCalculations import * 

from fChannel import *

from fAlgorithm import*

from fPlot import*

from tqdm import tqdm

'''
main.py
Written by: Refat khan
Last modified on september, 27, 2024
'''

sys_param = myf_sys_param()

Num_samples = 10000

x_axis_name = "EsoverNO_dB_cand"

if(x_axis_name == "EsoverNO_dB_cand"):

    x_axis_cand = np.arange(start=-10, stop=35, step=5)

Num_algorithms = 7

constellation = myf_constellation(sys_param)

Num_errors_bit = np.zeros((Num_algorithms,np.size(x_axis_cand,axis=0)))

Num_bits_eff = np.zeros((Num_algorithms,np.size(x_axis_cand,axis=0)))

for ind1 in tqdm(range(0,Num_samples)):

    # Step (1): Generate random channel

    param_channel = {}

    param_channel["seed_seq"] = Num_samples + ind1

    channel = myf_channel(sys_param, param_channel)

    # Step (2): Generate a random index of the constellation

    param_symbol_indices = {}

    param_symbol_indices["seed_seq"] = Num_samples + 1 + ind1

    symbol_indices = myf_symbol_indices(sys_param, param_symbol_indices)

     # Step (3): Encoding i.e., symbol_index -> constellation symbol (note: E[|x|^2]=2)

    x_vec = myf_x_vec(sys_param, constellation, symbol_indices)

    # print("x_vce : ",x_vec)

    #cereating matrix for total numbers of error and effective bit

    Num_errors_bit_temp = np.zeros((Num_algorithms, np.size(x_axis_cand, axis=0)))

    Num_bits_eff_temp = np.zeros((Num_algorithms, np.size(x_axis_cand, axis=0)))
  

    for ind2 in range(0, np.size(x_axis_cand, axis=0)):

        if (x_axis_name == "EsoverNO_dB_cand"):

            sys_param["EsoverNO_dB"] = x_axis_cand[ind2] # Es/N0 [dB]

            sys_param["EsoverNO"] = 10**(sys_param["EsoverNO_dB"]/10) # [no unit]

            sys_param["NO"] = sys_param["Es"]/sys_param["EsoverNO"] # Noise energy [Joule]


        P_mat, G_mat, Ns_opt = myf_P(sys_param)

        s_vec = P_mat @ x_vec

        # Step (5): Received symbol vector i.e., y_vec = H_mat*s_vec + z_vec

        param_y_vec = {}

        param_y_vec["seed_seq"] = Num_samples + 2 + ind1

        param_y_vec["s_vec"] = s_vec

        y_vec = myf_y_vec(sys_param,channel,param_y_vec)

        
        ## Algorithm 1 : Transmit matched Filter ########

        solutions_algorithm_1 = myf_algorithm_2(sys_param,channel,G_mat, mode="TxMF")

        s_vec = solutions_algorithm_1["P_mat"] @ x_vec

        param_y_vec = {}

        param_y_vec["seed_seq"] = Num_samples + 2 + ind1

        param_y_vec["s_vec"] = s_vec

        y_vec = myf_y_vec(sys_param,channel,param_y_vec)

        solutions = {}

        solutions["G_mat"] = G_mat

        x_vec_est = myf_x_vec_est(solutions,y_vec)

        param_Num_errors = {}

        param_Num_errors["symbol_indices"] = symbol_indices

        param_Num_errors["Ns_opt"] = sys_param["Ns"]

        param_Num_errors["y_vec"] = y_vec

        param_Num_errors["x_vec_est"] = x_vec_est

        

        Num_errors = myf_Num_errors(sys_param, constellation, param_Num_errors)

        Num_errors_bit_temp[0][ind2] = Num_errors["Num_bit_errors"]

        Num_bits_eff_temp[0][ind2] = Ns_opt * np.log2(sys_param["constellation_size"])

        ## Algorithm 2: transmit zero forcing filter ########

        solutions_algorithm_2 = myf_algorithm_2(sys_param,channel,G_mat, mode="TxZF")

        s_vec = solutions_algorithm_2["P_mat"] @ x_vec

        param_y_vec = {}

        param_y_vec["seed_seq"] = Num_samples + 2 + ind1

        param_y_vec["s_vec"] = s_vec

        y_vec = myf_y_vec(sys_param,channel,param_y_vec)

        solutions = {}

        solutions["G_mat"] = G_mat

        x_vec_est = myf_x_vec_est(solutions,y_vec)

        param_Num_errors = {}

        param_Num_errors["symbol_indices"] = symbol_indices

        param_Num_errors["Ns_opt"] = sys_param["Ns"]

        param_Num_errors["y_vec"] = y_vec

        param_Num_errors["x_vec_est"] = x_vec_est

        Num_errors = myf_Num_errors(sys_param, constellation, param_Num_errors)

        Num_errors_bit_temp[1][ind2] = Num_errors["Num_bit_errors"]

        Num_bits_eff_temp[1][ind2] = Ns_opt * np.log2(sys_param["constellation_size"])

        ## Algorithm 3: Transmit Wiener Filter ##########

        solutions_algorithm_3 = myf_algorithm_2(sys_param,channel,G_mat, mode="TxWF")

        s_vec = solutions_algorithm_3["P_mat"] @ x_vec

        param_y_vec = {}

        param_y_vec["seed_seq"] = Num_samples + 2 + ind1

        param_y_vec["s_vec"] = s_vec

        y_vec = myf_y_vec(sys_param,channel,param_y_vec)

        G_tx = G_mat

        solutions = {}

        solutions["G_mat"] = G_mat

        x_vec_est = myf_x_vec_est(solutions,y_vec)

        param_Num_errors = {}

        param_Num_errors["symbol_indices"] = symbol_indices

        param_Num_errors["Ns_opt"] = sys_param["Ns"]

        param_Num_errors["y_vec"] = y_vec

        param_Num_errors["x_vec_est"] = x_vec_est

        Num_errors = myf_Num_errors(sys_param, constellation, param_Num_errors)

        Num_errors_bit_temp[2][ind2] = Num_errors["Num_bit_errors"]

        Num_bits_eff_temp[2][ind2] = Ns_opt * np.log2(sys_param["constellation_size"])

        ##### Algorithm 4 : TxCMMSE When E_tr = 2 ##########

        target_value = 2 

        optimal_lambda = myf_find_optimal_lambda(sys_param, channel, G_mat, target_value, lambda_min=10**(-6), lambda_max=10**2, tolerance=1e-6)

        solutions_algorithm_4 = myf_algorithm_5(sys_param,channel,G_mat,optimal_lambda)

        s_vec = solutions_algorithm_4["P_mat"] @ x_vec

        param_y_vec = {}

        param_y_vec["seed_seq"] = Num_samples + 2 + ind1

        param_y_vec["s_vec"] = s_vec

        y_vec = myf_y_vec(sys_param,channel,param_y_vec)

        G_tx = G_mat

        solutions = {}

        solutions["G_mat"] = G_mat

        x_vec_est = myf_x_vec_est(solutions,y_vec)

        param_Num_errors = {}

        param_Num_errors["symbol_indices"] = symbol_indices

        param_Num_errors["Ns_opt"] = sys_param["Ns"]

        param_Num_errors["y_vec"] = y_vec

        param_Num_errors["x_vec_est"] = x_vec_est

        Num_errors = myf_Num_errors(sys_param, constellation, param_Num_errors)

        Num_errors_bit_temp[3][ind2] = Num_errors["Num_bit_errors"]

        Num_bits_eff_temp[3][ind2] = Ns_opt * np.log2(sys_param["constellation_size"])


        ###### Algorithm 5: TxWF when khi = 9 dB #####

        E_tr = 2

        Khi_inverse_dB = 9 ##dB

        Khi_dB = -Khi_inverse_dB ##dB

        Khi = 10**(Khi_dB/10) ## no unit

        solutions_algorithm_5 = myf_algorithm_6(sys_param,channel,G_mat, E_tr, Khi)

        s_vec = solutions_algorithm_5["P_mat"] @ x_vec

        param_y_vec = {}

        param_y_vec["seed_seq"] = Num_samples + 2 + ind1

        param_y_vec["s_vec"] = s_vec

        y_vec = myf_y_vec(sys_param,channel,param_y_vec)

        G_tx = G_mat

        solutions = {}

        solutions["G_mat"] = G_mat

        x_vec_est = myf_x_vec_est(solutions,y_vec)

        param_Num_errors = {}

        param_Num_errors["symbol_indices"] = symbol_indices

        param_Num_errors["Ns_opt"] = sys_param["Ns"]

        param_Num_errors["y_vec"] = y_vec

        param_Num_errors["x_vec_est"] = x_vec_est

        Num_errors = myf_Num_errors(sys_param, constellation, param_Num_errors)

        Num_errors_bit_temp[4][ind2] = Num_errors["Num_bit_errors"]

        Num_bits_eff_temp[4][ind2] = Ns_opt * np.log2(sys_param["constellation_size"])


        ##### Algorithm 6 : TxCMMSE When E_tr = 20 ##########

        target_value = 20 

        optimal_lambda = myf_find_optimal_lambda(sys_param, channel, G_mat, target_value, lambda_min=10**(-6), lambda_max=10**2, tolerance=1e-6)

        solutions_algorithm_6 = myf_algorithm_5(sys_param,channel,G_mat,optimal_lambda)

        s_vec = solutions_algorithm_6["P_mat"] @ x_vec

        param_y_vec = {}

        param_y_vec["seed_seq"] = Num_samples + 2 + ind1

        param_y_vec["s_vec"] = s_vec

        y_vec = myf_y_vec(sys_param,channel,param_y_vec)

        G_tx = G_mat

        solutions = {}

        solutions["G_mat"] = G_mat

        x_vec_est = myf_x_vec_est(solutions,y_vec)

        param_Num_errors = {}

        param_Num_errors["symbol_indices"] = symbol_indices

        param_Num_errors["Ns_opt"] = sys_param["Ns"]

        param_Num_errors["y_vec"] = y_vec

        param_Num_errors["x_vec_est"] = x_vec_est

        Num_errors = myf_Num_errors(sys_param, constellation, param_Num_errors)

        Num_errors_bit_temp[5][ind2] = Num_errors["Num_bit_errors"]

        Num_bits_eff_temp[5][ind2] = Ns_opt * np.log2(sys_param["constellation_size"])


        ### Algorihtm 7 : TxWF when khi = 20 dB #####

        E_tr = 2

        Khi_inverse_dB = 20.5 ##dB

        Khi_dB = -Khi_inverse_dB ##dB

        Khi = 10**(Khi_dB/10) ## no unit

        solutions_algorithm_7 = myf_algorithm_6(sys_param,channel,G_mat, E_tr, Khi)

        s_vec = solutions_algorithm_7["P_mat"] @ x_vec

        param_y_vec = {}

        param_y_vec["seed_seq"] = Num_samples + 2 + ind1

        param_y_vec["s_vec"] = s_vec

        y_vec = myf_y_vec(sys_param,channel,param_y_vec)

        G_tx = G_mat

        solutions = {}

        solutions["G_mat"] = G_mat

        x_vec_est = myf_x_vec_est(solutions,y_vec)

        param_Num_errors = {}

        param_Num_errors["symbol_indices"] = symbol_indices

        param_Num_errors["Ns_opt"] = sys_param["Ns"]

        param_Num_errors["y_vec"] = y_vec

        param_Num_errors["x_vec_est"] = x_vec_est

        Num_errors = myf_Num_errors(sys_param, constellation, param_Num_errors)

        Num_errors_bit_temp[6][ind2] = Num_errors["Num_bit_errors"]

        Num_bits_eff_temp[6][ind2] = Ns_opt * np.log2(sys_param["constellation_size"])

        

        


        

    ##Accumulation

    Num_errors_bit = Num_errors_bit + Num_errors_bit_temp

    Num_bits_eff = Num_bits_eff + Num_bits_eff_temp

BER = np.divide(Num_errors_bit,Num_bits_eff)

myf_plot_BER_Tx(sys_param,x_axis_cand,BER,x_axis_name)





        

        
    
                                   


    





                             




