import numpy as np

'''
fparam.py
Written by: Refat khan
Last modified on september, 27, 2024
'''

def myf_sys_param():

    sys_param = {}

    sys_param["Nt"] = 2

    sys_param["Nr"] = 2

    sys_param["Ns"] = min(sys_param["Nt"],sys_param["Nr"])

    sys_param["Es"] = 2

    # sys_param["P_rx"] = np.eye(2)

    # sys_param["G_tx"] = np.eye(2)

    sys_param["EsoverNO_dB"] = 5 # Es/N0 [dB]

    sys_param["EsoverNO"] = 10**(sys_param["EsoverNO_dB"]/10) # [no unit]

    sys_param["NO"] = sys_param["Es"]/sys_param["EsoverNO"] # Noise energy [Joule]

    sys_param["Rs"] = np.eye(sys_param["Ns"], dtype=np.float32)

    # sys_param["Rn"] = sys_param["NO"] * np.eye(2)

    sys_param["constellation_type"] = "QPSK"

    if(sys_param["constellation_type"] == "BPSK"):

        sys_param["constellation_size"] =2

    elif(sys_param["constellation_type"] == "QPSK"):

        sys_param["constellation_size"] = 4

    return sys_param