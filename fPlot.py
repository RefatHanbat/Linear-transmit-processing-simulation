import numpy as np

import matplotlib.pyplot as plt

'''
fplot.py
Written by: Refat khan
Last modified on september, 27, 2024
'''
def myf_plot_BER(sys_param, x_axis_cand, BER, x_axis_name):

    ###Parameterer##

    ##Functions##

    print("Modulations type : ",sys_param["constellation_type"] )
    print("Inner function : ", BER)

    fig, axes = plt.subplots(1,1)

    axes.semilogy(x_axis_cand, BER[0], 'b+-', markersize=12, markerfacecolor = "None", label=r'- RxMF')

    axes.semilogy(x_axis_cand, BER[1], 'rx-', markersize=12, markerfacecolor = "None", label=r'- RxZF')

    axes.semilogy(x_axis_cand, BER[2], 'k1-', markersize=12, markerfacecolor = "None", label=r'- RxMMSE')

    axes.semilogy(x_axis_cand, BER[3], 'b^-', markersize=12, markerfacecolor = "None", label=r'- TxMF')
    
    axes.semilogy(x_axis_cand, BER[4], 'rv-', markersize=12, markerfacecolor = "None", label=r'- TxZF')

    axes.semilogy(x_axis_cand, BER[5], 'k<-', markersize=12, markerfacecolor = "None", label=r'TxWF')

    

    axes.grid()

    if(x_axis_name == "EsoverNO_dB_cand"):

        axes.legend(loc="best")

        axes.set_xlabel(r'SNR, $\frac{E_s}{N_O} $dB')

    elif(x_axis_name == "EboverNO_dB_cand"):

        axes.legend(loc="best")

        axes.set_xlabel(r'SNR, $\frac{E_b}{N_O}$[dB]')

    axes.set_ylabel(r'BER')

    axes.set_xlim(min(x_axis_cand),max(x_axis_cand))

    axes.set_ylim(10**(-5),1)

    plt.savefig("fig_BER.png",dpi=300)

    plt.show()