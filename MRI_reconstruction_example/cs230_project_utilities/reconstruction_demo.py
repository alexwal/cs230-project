# -*- coding: utf-8 -*-
'''
CS230 Project Utilities
Alex Walczak, Avanika Narayan, Jordan Greenberg
'''
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

from . import signal_processing, plot


def reconstruct_image_from_fully_sampled_fourier_transform(image):
    '''
    (In MRI_reconstruction_example.ipynb.)
    '''
    
    # Forward Fourier Transform
    fft = signal_processing.fft_2D_centered(image)
    
    # Reconstruct original image from inverse Fourier Transform
    image_reconstruction = signal_processing.ifft_2D_centered(fft)
    
    # Compute peak signal-to-noise ratio
    PSNR = signal_processing.PSNR(np.abs(image_reconstruction), image, max_value=1.0) # may need to adjust max_value for different data
    print('PSNR of reconstrution: {}'.format(PSNR))
    
    # Plot results
    plt.subplot(3, 2, 1)
    plt.title('Original image')
    plot.imshowgray(np.abs(image)) # np.abs() because image may be complex-valued

    plt.subplot(3, 2, 2)
    plt.title('FFT')
    plot.imshowfft(fft)

    plt.subplot(3, 2, 3)
    plt.title('Reconstruction')
    plot.imshowgray(np.abs(image_reconstruction))

    plt.subplot(3, 2, 5)
    plt.title('Error in reconstruction')
    square_error = np.square(np.abs(image) - np.abs(image_reconstruction))
    plot.imshowgray(square_error)
    
def reconstruct_image_from_undersampled_fourier_transform(image, mask, pdf):
    '''
    Undersamples the FFT of `image` by masking with `mask` having probability density function `pdf` and
    visualizes the results.
    '''
    # Forward Fourier Transform
    original_fft = signal_processing.fft_2D_centered(image)
    
    # Undersample Fourier Transform and divide by PDF for density compensation
    masked_fft = (mask * original_fft) / pdf
    
    # Reconstruct image from undersampled Fourier Transform by compute the inverse Fourier Transform
    image_reconstruction = signal_processing.ifft_2D_centered(masked_fft)
    
    # Compute peak signal-to-noise ratio
    PSNR = signal_processing.PSNR(np.abs(image_reconstruction), image, max_value=1.0) # may need to adjust max_value for different data
    print('PSNR of reconstrution: {}'.format(PSNR))
    
    print('Mask preserves {0:.2f}% of FFT coefficients'.format(100.0 * mask.sum() / np.product(mask.shape)))

    plt.subplot(3, 2, 1)
    plt.title('Original image')
    plot.imshowgray(np.abs(image)) # np.abs() because image may be complex-valued

    plt.subplot(3, 2, 2)
    plt.title('FFT')
    plot.imshowfft(original_fft)

    plt.subplot(3, 2, 3)
    plt.title('Reconstruction')
    plot.imshowgray(np.abs(image_reconstruction))

    plt.subplot(3, 2, 4)
    plt.title('Mask')
    plot.imshowgray(mask)

    plt.subplot(3, 2, 5)
    plt.title('Error in reconstruction')
    square_error = np.square(np.abs(image) - np.abs(image_reconstruction))
    plot.imshowgray(square_error)

    plt.subplot(3, 2, 6)
    plt.title('Sampled FFT')
    plot.imshowfft(masked_fft)
