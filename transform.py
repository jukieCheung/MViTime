import numpy as np
from scipy import signal
import cv2

import random
class Transform:
    def __init__(self):
        pass

    def permute(self,signal, pieces):
        """
        signal: numpy array (batch x window)
        pieces: number of segments along time
        """
        signal = signal.T
        pieces = int(np.ceil(np.shape(signal)[0] / (np.shape(signal)[0] // pieces)).tolist()) #向上取整
        piece_length = int(np.shape(signal)[0] // pieces)

        sequence = list(range(0, pieces))
        np.random.shuffle(sequence)

        permuted_signal = np.reshape(signal[:(np.shape(signal)[0] // pieces * pieces)],
                                     (pieces, piece_length)).tolist()

        tail = signal[(np.shape(signal)[0] // pieces * pieces):]
        permuted_signal = np.asarray(permuted_signal)[sequence]
        permuted_signal = np.concatenate(permuted_signal, axis=0)
        permuted_signal = np.concatenate((permuted_signal,tail[:,0]), axis=0)
        permuted_signal = permuted_signal[:,None]
        permuted_signal = permuted_signal.T
        return permuted_signal


    def crop_resize(self, signal, size):
        signal = signal.T
        size = signal.shape[0] * size
        size = int(size)
        start = random.randint(0, signal.shape[0]-size)
        crop_signal = signal[start:start + size,:]
        # print(crop_signal.shape)

        crop_signal = cv2.resize(crop_signal, (1, 3072), interpolation=cv2.INTER_LINEAR)
        # print(crop_signal.shape)
        crop_signal = crop_signal.T
        return crop_signal

