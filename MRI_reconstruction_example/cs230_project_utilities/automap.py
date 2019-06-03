# -*- coding: utf-8 -*-
'''
CS230 Utilities for data from Wintermark lab
Alex Walczak, Avanika Narayan, Jordan Greenberg
'''
from __future__ import division
import numpy as np
import os, glob
import h5py # for loading .h5 files
import matplotlib.pyplot as plt

# Handle .h5 files

def inspect_h5_file(path):
    with h5py.File(path, 'r') as f:
        # Looks like .h5 files are like dictionaries
        print('Found keys: {}'.format(f.keys()))
        
        # Let's find all the nested keys
        print('\nFinding nested keys...')
        show_keys_for_h5_file(path)
        
        # Unwrap the data stored in the keys
        # known keys: 'clasificacion', 'images', 'kspace/mag', 'kspace/phase'
        print('\nUnwrapping data at known keys...')
        classification = f['clasificacion']
        images = f['images']
        magnitude = f['kspace/mag']
        phase = f['kspace/phase']
        print(classification)
        print(images)
        print(magnitude)
        print(phase)

def show_keys_for_h5_file(path):
    '''
    Convenience function for printing the nested keys in an
    .h5 file.
    '''
    def print_nested_keys(key, value):
        print(key)
        for k, v in value.attrs.items():
            print('{}: {}'.format(key, val))

    with h5py.File(path, 'r') as f:
        f.visititems(print_nested_keys)
        
def read_h5_file(path):
    '''
    Reads a .h5 file from `path`, unwrapping data stored in the known
    keys: 'clasificacion', 'images', 'kspace/mag', 'kspace/phase'.
    '''
    with h5py.File(path, 'r') as f:
        classification = f['clasificacion'].value
        images = f['images'].value
        magnitude = f['kspace/mag'].value
        phase = f['kspace/phase'].value
        
    # Convert to float32 and scale data to [0, 1]
    images = images.astype(np.float32)
    images -= images.min(axis=(1, 2), keepdims=True)
    images /= images.max(axis=(1, 2), keepdims=True)
    
    # Additional conversion scaling here with magnitude data.
    # Most ML and scientific frameworks expect data to be in [0, 1].
    magnitude = magnitude.astype(np.float32)
    magnitude -= magnitude.min(axis=(1, 2), keepdims=True)
    magnitude /= magnitude.max(axis=(1, 2), keepdims=True)
    
    phase = phase.astype(np.float32)
    # Don't scale phase (it represents an angle, so it's always in [0, pi])
    
    return {'classification': classification,
            'images': images,
            'magnitude': magnitude,
            'phase': phase}
    
# Handle .txt files

def inspect_text_file(path):
    with open(path, 'rb') as f:
        print(f.read())

def find_dataset_files(dataset_directory):
    # Find .h5 and .txt files in `dataset_directory`
    # Assuming at most one name.h5 and one name.txt file in `dataset_directory`.
    h5_file_pattern = os.path.join(dataset_directory, '*.h5')
    text_file_pattern = os.path.join(dataset_directory, '*.txt')
    h5_paths = glob.glob(h5_file_pattern)
    text_paths = glob.glob(text_file_pattern)
    print('Found {} .h5 files and {} .txt files.'.format(len(h5_paths), len(text_paths)))
    
    # Merge .h5 and .txt paths into one data structure: a dict where key is filename
    # without extension and value is dict {'h5': h5_path, 'text': text_path}.
    get_filename = lambda path: os.path.splitext(os.path.basename(path))[0]
    files = {}
    for path in h5_paths:
        filename = get_filename(path)
        files[filename] = {'h5': path}
    for path in text_paths:
        filename = get_filename(path)
        files[filename]['text'] = path
      
    return files
