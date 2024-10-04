import time

import numpy as np

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


def myf_y_vec(sys_param, channel, param_rand_noise, s_vec):      # receive signal function

    # Parameters #

    RxAnt = sys_param["Nr"]

    No = sys_param["NO"]

    H_mat = channel["H_mat"]

    seed_seq = param_rand_noise["seed_seq"]


    # Function #

    rng = np.random.default_rng(seed=seed_seq)
    z_vec = rng.normal(loc=0, scale=np.sqrt(No/2), size=(RxAnt, 1)) + 1j * rng.normal(loc=0, scale=np.sqrt(No/2), size=(RxAnt, 1))
    y_vec = H_mat@s_vec + z_vec
    #print("y_vec is = ", y_vec)

    return y_vec

def myf_x_vec_est(solutions,y_vec):

    x_vec_est = solutions["G_mat"] @ y_vec

    return x_vec_est


def myf_Num_errors(sys_param, constellation, x_vec_est, symbol_indices, Ns_opt):

    # Parameters #

    constellation_type = sys_param["constellation_type"]
    constellation_bits = constellation["constellation_bits"]

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
        
            # This is just find out error bit
            Num_bit_errors = Num_bit_errors + np.sum(np.abs(constellation_bits[int(symbol_indices[ind])] - \
                                                            constellation_bits[int(symbol_indices_dec[ind])]))
            # print(" bit errors =", bit_errors)

    Num_errors = {}
    Num_errors["Num_symbol_errors"] = Num_symbol_errors
    Num_errors["Num_bit_errors"] = Num_bit_errors

    return Num_errors

     



# def myf_Num_errors(sys_param,constellation,x_vec_est,param_Num_errors):

#     constellation_type = sys_param["constellation_type"]

#     constellation_bits = constellation["constellation_bits"]

#     symbol_indices = param_Num_errors["symbol_indices"]

#     Ns_opt = param_Num_errors["Ns_opt"]

#     symbol_indices_dec = np.zeros(Ns_opt)

#     for ind in range(0, Ns_opt):

#         if (constellation_type=="BPSK"):

#             if(np.real(x_vec_est[ind][0]) < 0 ):

#                 symbol_indices_dec[ind] = 0

#             elif (np.real(x_vec_est[ind][0])>=0):

#                 symbol_indices_dec[ind] = 1

#         elif (constellation_type == "QPSK"):

#             if((np.real(x_vec_est[ind][0])< 0 ) and ((np.imag(x_vec_est[ind][0]) < 0))):

#                 symbol_indices_dec[ind] = 0

#             elif((np.real(x_vec_est[ind][0])<0) and ((np.imag(x_vec_est[ind][0])>=0))):

#                 symbol_indices_dec[ind] = 1

#             elif ((np.real(x_vec_est[ind][0]) >= 0) and ((np.imag(x_vec_est[ind][0]) < 0))):

#                 symbol_indices_dec[ind] = 2

#             elif ((np.real(x_vec_est[ind][0]) >= 0) and ((np.imag(x_vec_est[ind][0]) >= 0))):

#                 symbol_indices_dec[ind] = 3

#     Num_errors_symbol = 0

#     Num_errors_bit = 0

#     for ind in range(0,Ns_opt):

#         if(symbol_indices[ind] != symbol_indices_dec[ind]):

#             Num_errors_symbol = Num_errors_symbol + 1

#             Num_errors_bit = Num_errors_bit + np.sum(np.abs(constellation_bits[int(symbol_indices[ind])]\
                                                            
#                             - constellation_bits[int(symbol_indices_dec[ind])]))
#     Num_errors = {}

#     Num_errors["Num_errors_symbol"] = Num_errors_symbol

#     Num_errors["Num_errors_bit"] = Num_errors_bit
            
#     return Num_errors

    

        

