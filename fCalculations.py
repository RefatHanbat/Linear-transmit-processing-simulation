import time

import numpy as np

from scipy.linalg import sqrtm

'''
fCalculations.py
Written by: Refat khan
Last modified on september, 27, 2024
'''

def myf_constellation(sys_param):

    # Parameters #
    constellation_type = sys_param["constellation_type"]

    # Function #
    E_symbol_avg = 2  # unit average symbol energy

    constellation = {}

    if(constellation_type == "BPSK"):
        constellation_symbols = np.sqrt(E_symbol_avg/1)*np.array([[-1], [+1]])
        constellation_bits = np.array([[0], [1]])
    elif (constellation_type == "QPSK"):
        constellation_symbols = np.sqrt(E_symbol_avg/2)*np.array([[(-1) + 1j*(-1)], [(-1) + 1j*(+1)],\
                                                                  [(+1) + 1j*(-1)], [(+1) + 1j*(+1)]])
        constellation_bits = np.array([[0, 0], [0, 1],\
                                       [1, 0], [1, 1]])
    constellation["constellation_symbols"] = constellation_symbols
    constellation["constellation_bits"] = constellation_bits

    return constellation

def myf_symbol_indices(sys_param, param_symbol_indices):

    # Parameters #
    Ns = sys_param["Ns"]
    constellation_size = sys_param["constellation_size"]
    seed_seq = param_symbol_indices["seed_seq"]

    # Function #
    rng = np.random.default_rng(seed=seed_seq)
    symbol_indices = rng.integers(low=0, high=constellation_size, size=Ns)

    return symbol_indices

def myf_x_vec(sys_param, constellation, symbol_indices):

    # Parameters #
    Ns = sys_param["Ns"]
    constellation_symbols = constellation["constellation_symbols"]

    # Function #
    x_vec = np.zeros((Ns, 1), dtype=np.complex64)
    for ind in range(Ns):
         x_vec[ind][0] = constellation_symbols[symbol_indices[ind]][0]
 
    return x_vec


def myf_y_vec(sys_param,channel,param_y_vec):
     # receive signal function

    # Parameters #

    Nr = sys_param["Nr"]

    No = sys_param["NO"]

    H_mat = channel["H_mat"]

    seed_seq = param_y_vec["seed_seq"]

    s_vec = param_y_vec["s_vec"]


    # Function #

    rng = np.random.default_rng(seed=seed_seq)

    z_vec = rng.normal(loc=0, scale=np.sqrt(No/2), size=(Nr, 1)) + 1j * rng.normal(loc=0, scale=np.sqrt(No/2), size=(Nr, 1))

    y_vec = H_mat@s_vec + z_vec


    return y_vec

def myf_x_vec_est(solutions,y_vec):

    x_vec_est = solutions["G_mat"] @ y_vec

    return x_vec_est


def myf_Num_errors(sys_param, constellation, param_Num_errors):

    # Parameters #

    constellation_type = sys_param["constellation_type"]

    constellation_bits = constellation["constellation_bits"]

    Ns_opt = param_Num_errors["Ns_opt"]

    x_vec_est = param_Num_errors["x_vec_est"]

    symbol_indices = param_Num_errors["symbol_indices"]



    # Function #

    symbol_indices_dec = np.zeros(Ns_opt)

    for ind in range(0, Ns_opt):

        if (constellation_type=="BPSK"):

            if(np.real(x_vec_est[ind][0])<0):

                symbol_indices_dec[ind] = 0

            elif (np.real(x_vec_est[ind][0])>=0):

                symbol_indices_dec[ind] = 1

        elif (constellation_type == "QPSK"):

            if((np.real(x_vec_est[ind][0])<0) and (np.imag(x_vec_est[ind][0])<0)):

                symbol_indices_dec[ind] = 0

            elif((np.real(x_vec_est[ind][0])<0) and (np.imag(x_vec_est[ind][0])>=0)):

                symbol_indices_dec[ind] = 1

            elif ((np.real(x_vec_est[ind][0]) >= 0) and (np.imag(x_vec_est[ind][0]) < 0)):

                symbol_indices_dec[ind] = 2

            elif ((np.real(x_vec_est[ind][0]) >= 0) and (np.imag(x_vec_est[ind][0]) >= 0)):

                symbol_indices_dec[ind] = 3

    # Counting errors
    Num_symbol_errors = 0

    Num_bit_errors = 0

    for ind in range(0, Ns_opt):

        if (symbol_indices[ind] != symbol_indices_dec[ind]):

            Num_symbol_errors = Num_symbol_errors + 1   # counting number of symbol error
        
            # Counting bit errors 

            Num_bit_errors = Num_bit_errors + np.sum(np.abs(constellation_bits[int(symbol_indices[ind])] - \
                                                            constellation_bits[int(symbol_indices_dec[ind])]))
            

    Num_errors = {}
    Num_errors["Num_symbol_errors"] = Num_symbol_errors
    Num_errors["Num_bit_errors"] = Num_bit_errors

    return Num_errors

def myf_J_rx(sys_param,channel,P_mat):

    Rs = sys_param["Rs"] 

    Ns = sys_param["Ns"]

    H_mat = channel["H_mat"] 

    H_mat_H = np.transpose(np.conjugate(H_mat))

    P_mat = P_mat

    P_mat_H = np.transpose(np.conjugate(P_mat))

    Rn = sys_param["NO"] * np.eye(Ns, dtype=np.float32)

    Rn_inverse = np.linalg.inv(Rn)

    Rs_sqrt = sqrtm(Rs)

    Rs_sqrt_H = np.transpose(np.conjugate(Rs_sqrt))

    J_mat = Rs_sqrt_H @ P_mat_H @ H_mat_H @ Rn_inverse @ H_mat @ P_mat @ Rs_sqrt

    solution = {}

    solution["J_mat"] = J_mat

    return solution

def myf_J_tx(sys_param,channel,G_mat):

    E_tr = sys_param["Es"]

    Ns = sys_param["Ns"]

    G_mat = G_mat

    G_mat_H = np.transpose(np.conjugate(G_mat))

    H_mat = channel["H_mat"]

    Rn = sys_param["NO"] * np.eye(Ns, dtype=np.float32)

    H_mat_H = np.transpose(np.conjugate(H_mat))

    J_mat = (G_mat @ H_mat @ H_mat_H @ G_mat_H) * (np.divide(E_tr,np.trace(G_mat @ Rn @ G_mat_H)))


    solutions = {}

    solutions["J_mat"] = J_mat

    return solutions

import numpy as np

def myf_compute_trace(H_mat, G_mat, Rs, Ns, lambda_val):
    
    H_mat_H = np.transpose(np.conjugate(H_mat))  

    G_mat_H = np.transpose(np.conjugate(G_mat))  

    I_N = np.eye(Ns, dtype=H_mat.dtype)  

    
    inverse_term = np.linalg.inv(H_mat_H @ G_mat_H @ G_mat @ H_mat + lambda_val * I_N)

    
    squared_inverse = inverse_term @ inverse_term

   
    trace_value = np.abs(np.trace(squared_inverse @ H_mat_H @ G_mat_H @ Rs @ G_mat @ G_mat_H))

    return trace_value

def myf_find_optimal_lambda(sys_param, channel, G_mat, target_value, lambda_min=1e-6, lambda_max=1e2, tolerance=1e-6):
    
    Rs = sys_param["Rs"]

    H_mat = channel["H_mat"]

    Ns = sys_param["Ns"]

    
    while lambda_max - lambda_min > tolerance:

        lambda_mid = (lambda_min + lambda_max) / 2.0

        
        func_value = myf_compute_trace(H_mat, G_mat, Rs, Ns, lambda_mid)

        # print(f"λ = {lambda_mid:.6f}, Trace = {func_value:.6f}")

        
        if func_value > target_value:

            lambda_min = lambda_mid  # Move the lower bound up

        else:

            lambda_max = lambda_mid  # Move the upper bound down

    
    optimal_lambda = (lambda_min + lambda_max) / 2.0

    # print(f"Optimal λ found: {optimal_lambda:.6f}")

    return optimal_lambda




    




    

        

