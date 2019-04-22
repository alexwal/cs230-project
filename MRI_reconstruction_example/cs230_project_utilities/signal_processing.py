# -*- coding: utf-8 -*-
'''
CS230 Project Signal Processing Utilities
Alex Walczak, Avanika Narayan, Jordan Greenberg
'''
from __future__ import division
import numpy as np


def PSNR(A, B, max_value=256.0):
    '''
    Compute the peak signal-to-noise ratio (PSNR) of the inputs.
    
    Parameters
    ----------
    A : numpy.ndarray
        same shape as B, takes on values [0, max_value]
    B : numpy.ndarray
        same shape as A, takes on values [0, max_value]
    max_value : float
        maximum possible value of A and B (e.g., 255 for 8-bit images)
    
    Returns
    -------
    PSNR: float
        PSNR of A and B
        
    Examples
    --------
    >>> PSNR(np.array([0,0.001,2,3,4.0000001,5]), np.array([49,6,1.005,3,0,4]), max_value=50.0)
    7.860436497177595
    
    '''
    MSE = mean_square_error(A, B)
    PSNR = 10.0 * np.log10(max_value ** 2 / MSE)
    return PSNR

def mean_square_error(A, B):
    '''
    Compute the mean square error of the inputs.
    
    Parameters
    ----------
    A : numpy.ndarray
        same shape as B
    B : numpy.ndarray
        same shape as A
        
    Examples
    --------
    >>> mean_square_error(np.array([1, -2, 3]), np.array([-4, 5, 6]))
    27.666666666666668
    
    '''
    return np.mean(np.square(A.astype(float) - B.astype(float)))

def fft_2D_centered(x):
    '''
    Compute the 2D, centered Fourier transform of x
    
    Parameters
    ----------
    x : numpy.ndarray
        2-dimensional
    
    Examples
    --------
    >>> np.random.seed(0)
    >>> x = np.random.randn(10, 10)
    >>> y = utils.signal_processing.fft_2D_centered(x)
    >>> x_reconstructed = utils.signal_processing.ifft_2D_centered(y)
    >>> mean_square_error(x, x_reconstructed.real)
    9.659321080141895e-32
    
    '''
    return 1 / np.sqrt(np.prod(x.shape)) * np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))

def ifft_2D_centered(y):
    '''
    Compute the inverse Fourier transform of 2D, centered y
    
    Parameters
    ----------
    y : numpy.ndarray
        2-dimensional
    
    Notes
    -----
    See fft_2D_centered(x).
    
    '''
    return np.sqrt(np.prod(y.shape)) * np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(y)))

def gaussian_kernel_2D(m, n, sigma=1.0, mu=(0.0, 0.0), centered=True):
    '''
    # 2D Gaussian kernel, shape [m, n], standard deviation = sigma, mean = mu.
    # m: integer > 0
    # n: integer > 0
    # sigma: standard deviation
    # mu: mean
    # centered: if True, center of output kernel represents the origin (exactly when odd); otherwise,
    #   kernel[i, j] corresponds to [i, j] in Cartesian coordinates.
    # returns: gaussian kernel, shape [m, n]
    '''
    assert m > 1 and n > 1 and isinstance(m, int) and isinstance(n, int), 'm, n must be integers > 0'

    x0, y0 = mu
    if centered:
        # Index [m//2 + 1, n//2, + 1] corresponds to origin.
        x, y = np.arange(m) - int(m // 2), np.arange(n) - int(n // 2)
    else:
        # Index [i, j] corresponds to [i, j] in Cartesian coordinates.
        x, y = np.arange(m), np.arange(n)

    gx = np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
    gy = np.exp(-(y - y0) ** 2 / (2 * sigma ** 2))
    
    # x and y are independent, so PDF is outer product of x and y
    pdf = np.outer(gx, gy)
    pdf /= np.sum(pdf) # normalize so that sum(pdf) = 1

    return pdf

def uniform_mask(m, n, p):
    # Binary mask of shape (m, n)
    # mask[i, j, ...] == uniform(interval=[0, 1)) < p
    kernel = np.ones((m, n)) * p
    mask = np.float32(np.random.rand(m, n) < kernel)
    return mask, kernel

def gaussian_mask(m, n, p, sigma=1.0):
    kernel = gaussian_kernel_2D(m, n, sigma=sigma, mu=(0.0, 0.0), centered=True)
    g = kernel / kernel.max()
    R = np.random.rand(m, n)
    mask = (R < p) | (R < g)
    return mask, kernel
