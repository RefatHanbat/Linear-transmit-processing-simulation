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
