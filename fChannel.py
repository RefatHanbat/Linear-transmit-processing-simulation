import numpy as np
'''
fChannel.py
Written by: Refat khan
Last modified on september, 27, 2024
'''
def myf_channel(sys_param, param_rand_channel):

     # Parameters #
    TxAnt = sys_param["Nt"]
    RxAnt = sys_param["Nr"]
    seed_seq = param_rand_channel["seed_seq"]

    # Function #
    rng = np.random.default_rng(seed=seed_seq)  # Randomly channel generates

    channel = {}
    # Large-scale path-loss
    Pl = 1
    # Small-scale path-loss
    H = rng.normal(loc=0, scale=np.sqrt(1 / 2), size=(RxAnt, TxAnt)) \
        + 1j * rng.normal(loc=0, scale=np.sqrt(1 / 2), size=(RxAnt, TxAnt))

    # Final channel
    H_mat = np.sqrt(Pl) * H
    channel["H_mat"] = H_mat

    return channel
