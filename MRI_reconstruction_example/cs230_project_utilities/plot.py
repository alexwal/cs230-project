# -*- coding: utf-8 -*-
'''
CS230 Project Utilities
Alex Walczak, Avanika Narayan, Jordan Greenberg
'''

import numpy as np

from .wavelets import imshowgray, imshowWAV


def imshowfft(fft):
    imshowgray(20 * np.log10(np.abs(fft)))