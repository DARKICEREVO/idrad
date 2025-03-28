import numpy as np
import torch as th

import matplotlib.pylab as plt

# Utility function to convert radar data

def microdoppler_transform(sample, values=None, standard_scaling=False, minmax_scaling=False, local_scaling=False, preprocessing=False):

    if preprocessing:
        sample = np.concatenate((sample[:, 24:127], sample[:, 130:232]), axis=1)

    if minmax_scaling:
        sample = (sample - values["min"]) / (values["max"] - values["min"])

    if standard_scaling:
        sample = (sample - values["mean"]) / values["std"]

    if local_scaling:
        sample = (sample - sample.min()) / (sample.max() - sample.min())

    
    # print(sample.shape)
    # plt.imshow(sample.T)
    # plt.show()

    return th.from_numpy(np.expand_dims(np.transpose(sample.astype(np.float32)), axis=-1))

