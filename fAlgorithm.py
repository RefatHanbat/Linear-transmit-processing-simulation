import numpy as np

'''
fAlgorithms.py
Written by: Refat khan
Last modified on september, 27, 2024
'''
def myf_P(sys_param):

    Ns = sys_param["Ns"]

    P_mat = np.eye(Ns, dtype=np.float32 )

    G_mat = np.eye(Ns, dtype=np.float32 )

    return P_mat, G_mat, Ns


def myf_algorithm_1(sys_param,channel,P_mat, mode="RxMF"):

    Rs = sys_param["Rs"]

    Ns = sys_param["Ns"]

    Rs_inverse = np.linalg.inv(Rs)

    P = P_mat

    P_H = np.transpose(np.conjugate(P))

    H_mat = channel["H_mat"]

    H_mat_H = np.transpose(np.conjugate(H_mat))

    Rn = sys_param["NO"] * np.eye(Ns, dtype=np.float32)

    Rn_inverse = np.linalg.inv(Rn)


    if(mode=="RxMF"):
        
        G_mat = Rs@P_H@H_mat_H@ Rn_inverse

    elif(mode =="RxZF"):

        G_mat = np.linalg.inv(P_H @H_mat_H @ Rn_inverse @ H_mat @ P) @ P_H @ H_mat_H @ Rn_inverse

    elif(mode == "RxWF") :

        G_mat = np.linalg.inv((P_H @ H_mat_H @ Rn_inverse @ H_mat @ P + Rs_inverse)) @ P_H @ H_mat_H @ Rn_inverse

    solutions = {}

    solutions["G_mat"] = G_mat

    return solutions

def myf_algorithm_2(sys_param,channel,G_mat, mode="TxMF"):

    E_tr = sys_param["Es"]

    H_mat = channel["H_mat"]

    H_mat_H = np.transpose(np.conjugate(H_mat))

    Rs = sys_param["Rs"]

    G_tx = G_mat

    Ns = sys_param["Ns"]

    G_tx_H = np.transpose(np.conjugate(G_tx))

    Rn = sys_param["NO"] * np.eye(Ns, dtype=np.float32)

    if (mode=="TxMF") :

        beta_TxMF = np.sqrt((E_tr)/(np.trace(H_mat_H@G_tx_H@Rs@G_tx@H_mat)))

        P_mat = beta_TxMF * (H_mat_H@G_tx_H)

    elif(mode == "TxZF") :

        beta_TxZF = np.sqrt((E_tr)/(np.trace(np.linalg.inv(G_tx@H_mat@H_mat_H@G_tx_H)@Rs)))

        P_mat = beta_TxZF * (H_mat_H@G_tx_H@(np.linalg.inv(G_tx@H_mat@H_mat_H@G_tx_H)))

    elif(mode=="TxWF"):

        F = H_mat_H @ G_tx_H @ G_tx @ H_mat + np.divide(np.trace(G_tx@Rn@G_tx_H),E_tr) * np.eye(Ns, dtype=np.float32)

        F_inv = np.linalg.inv(F)

        F_inv_squared = F_inv @ F_inv

        beta_TxWF = np.sqrt(np.divide((E_tr),(np.trace(F_inv_squared@H_mat_H@G_tx_H@Rs@G_tx@H_mat))))

        P_mat = beta_TxWF * (F_inv@H_mat_H@G_tx_H)


    
    solutions = {}

    solutions["P_mat"] = P_mat

    return solutions

def myf_algorithm_3(sys_param,J_rx, mode="RxMF"):

    Rs = sys_param["Rs"]

    J_rx = J_rx["J_mat"]

    Ns = sys_param["Ns"]

    if (mode == "RxMF"):

        nominator = np.trace(J_rx @ Rs) * np.trace(J_rx @ Rs)

        denominator = np.trace((J_rx @ J_rx + J_rx)@Rs)

        epsilon_mat = np.divide(nominator,denominator)

        epsilon_mat = np.trace(Rs) - epsilon_mat

    elif(mode=="RxZF"):

        nominator = np.trace(Rs) * np.trace(Rs)

        denominator = np.trace(Rs) + np.trace(np.linalg.inv(J_rx)@Rs)

        epsilon_mat = np.trace(Rs) - np.divide(nominator,denominator)

    elif(mode == "RxWF"):

        right_part = np.trace(np.linalg.inv(J_rx + np.eye(Ns,dtype=np.float32))@ J_rx @Rs)

        epsilon_mat = np.trace(Rs) - right_part


    solutions = {}

    solutions["epsilon_mat"] = epsilon_mat

    return solutions

def myf_algorithm_4(sys_param,J_tx, mode="RxMF"):

    Rs = sys_param["Rs"]

    Ns = sys_param["Ns"]

    J_tx = J_tx["J_mat"]

    if(mode == "TxMF"):

        nominator = (np.trace(J_tx @ Rs) * np.trace(J_tx@ Rs)) 

        dominator = np.trace((J_tx @ J_tx + J_tx) @ Rs)

        epsilon_mat = np.trace(Rs) - np.divide(nominator,dominator)

    elif(mode == "TxZF"):

        nominator = np.trace(Rs) * np.trace(Rs)

        denominator = np.trace(Rs) + np.trace(np.linalg.inv(J_tx)@Rs)

        epsilon_mat = np.trace(Rs) - np.divide(nominator,denominator)


    elif(mode == "TxWF"):

        right_part = np.trace(np.linalg.inv((J_tx + np.eye(Ns, dtype=np.float32))) @ J_tx @ Rs)

        epsilon_mat = np.trace(Rs) - right_part

    solutions = {}

    solutions["epsilon_mat"] = epsilon_mat

    return solutions

def myf_algorithm_5(sys_param,channel,G_mat,optimal_lambda):

    Ns = sys_param["Ns"]

    H_mat = channel["H_mat"]

    H_mat_H = np.transpose(np.conjugate(H_mat))

    G_mat = G_mat

    G_mat_H = np.transpose(np.conjugate(G_mat))

    optimal_lambda = optimal_lambda


    I_N = np.eye(Ns, dtype=np.float32)

    inverse_part = np.linalg.inv(H_mat_H@G_mat_H@G_mat@H_mat + optimal_lambda * I_N)

    P_mat = inverse_part @ H_mat_H @ G_mat_H

    solutions = {}

    solutions["P_mat"] = P_mat

    return solutions

def myf_algorithm_6(sys_param,channel,G_mat, E_tr, Khi):

    Ns = sys_param["Ns"]

    Rs = sys_param["Rs"]

    E_tr = E_tr

    H_mat = channel["H_mat"]

    H_mat_H = np.transpose(np.conjugate(H_mat))

    G_tx = G_mat

    G_tx_H = np.transpose(np.conjugate(G_tx))

    Rn = sys_param["NO"] * np.eye(Ns, dtype=np.float32)

    I_N = np.eye(Ns, dtype=np.float32)

    Khi = Khi

    F = H_mat_H @ G_tx_H @ G_tx @ H_mat + Khi * I_N

    F_inv = np.linalg.inv(F)

    F_inv_squared = F_inv @ F_inv

    beta_TxWF = np.sqrt(np.divide((E_tr),(np.trace(F_inv_squared @ H_mat_H @ G_tx_H @Rs @G_tx @ H_mat))))

    P_mat = beta_TxWF * (F_inv @ H_mat_H @ G_tx_H)

    solutions = {}

    solutions["P_mat"] = P_mat

    return solutions












       


