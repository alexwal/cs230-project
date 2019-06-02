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

import tensorflow as tf


# Constants

FASTMRI_ORIGINAL_FFT_SHAPE = (640, 368)
FASTMRI_ORIGINAL_IMAGE_SHAPE = (320, 320)

# Utility functions for creating TFRecord datasets

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

# Raw data

def read_h5_file(path, coils):
    '''
    Reads a fastMRI .h5 file from `path`, unwrapping data stored in the known
    keys.
    '''
    assert coils in ('single', 'multi'), 'Coils must be either \'single\' or \'multi\'.'
    
    with h5py.File(path, 'r') as f:
        metadata = f['ismrmrd_header'].value
        kspace = f['kspace'].value
        reconstruction = f['reconstruction_esc'].value if coils == 'single' else f['reconstruction_rss'].value
    
    return {'coils': coils,
            'metadata': metadata,
            'kspace': kspace,
            'images': reconstruction}

# Convert to TFRecords

def _tf_example_for_slice_in_mri_volume(index, path, fft, image):
    
    assert len(fft.shape) > 2 and fft.shape[-1] == 2, 'FFT must have 2 channels representing real and imaginary values.'
    
    # Create list of bytes with .tostring() for multidimensional ndarrays.
    example = tf.train.Example(features=tf.train.Features(feature={
        'path': bytes_feature(path),
        'sequence_index': int64_feature(index),
        'fft': bytes_feature(fft.tostring()),
        'image': bytes_feature(image.tostring()),
        
        # save dimension of fft and image to restore these keys
        'image_dimension_0': int64_feature(image.shape[0]),
        'image_dimension_1': int64_feature(image.shape[1]),
        'fft_dimension_0': int64_feature(fft.shape[0]),
        'fft_dimension_1': int64_feature(fft.shape[1])
    }))
    
    return example

def _tf_examples_for_h5_file(path):
    
    h5_data = read_h5_file(path, coils='single')
    
    # Raw kspace data
    kspace_sequence = h5_data['kspace']
    real_sequence = kspace_sequence.real
    imaginary_sequence = kspace_sequence.imag
    
    fft_sequence = np.concatenate((np.expand_dims(real_sequence, axis=3),
                                   np.expand_dims(imaginary_sequence, axis=3)),
                                  axis=3)
    
    # Expected reconstruction images
    image_sequence = h5_data['images']
    image_sequence = np.expand_dims(image_sequence, axis=-1) # model expects 4 dims

    examples = []
    for i in range(len(image_sequence)):
        example = _tf_example_for_slice_in_mri_volume(i, path, fft_sequence[i], image_sequence[i])
        examples.append(example)
    
    return examples

def convert_fastmri_dataset_to_tfrecord_files(raw_data_directory, tfrecord_directory):
    tfrecord_file_pattern = os.path.join(tfrecord_directory, 'shard-{}.tfrecord')
    tfrecord_index = 0
    max_examples_per_tfrecord_file = 8#!!!512
    examples_in_current_tfrecord_file = 0

    os.makedirs(os.path.dirname(tfrecord_file_pattern), exist_ok=True)
    
    tfrecord_path = tfrecord_file_pattern.format(tfrecord_index)
    writer = tf.io.TFRecordWriter(tfrecord_path)
    
    h5_path_pattern = os.path.join(raw_data_directory, '*.h5')
    h5_paths = tf.data.Dataset.list_files(h5_path_pattern, shuffle=True)
    
    for path in h5_paths:
        path = path.numpy()
        try:
            examples = _tf_examples_for_h5_file(path)
            for example in examples:
                writer.write(example.SerializeToString())
                examples_in_current_tfrecord_file += 1
                
                if examples_in_current_tfrecord_file >= max_examples_per_tfrecord_file:
                    writer.close()
                    tfrecord_index += 1
                    tfrecord_path = tfrecord_file_pattern.format(tfrecord_index)
                    writer = tf.io.TFRecordWriter(tfrecord_path)
                    examples_in_current_tfrecord_file = 0
                
        except Exception as e:
            print('Error creating example: {}'.format(e))

# Load TFRecords

def load_dataset(data_locations, batch_size, shuffle_buffer_size, include_all_parsed_features,
                 ignore_errors, preprocessing_function=None):
    '''
    Returns iterator of fastMRI data located in `data_locations`.
    
    data_locations:  A string, a list of strings, or a `tf.Tensor` of string type
    (scalar or vector), representing the filename glob (i.e. shell wildcard)
    pattern(s) that will be matched.
    
    preprocessing_function: A function that takes in a parsed example and prepares it
    for use in a training or evaluation tf.data.Dataset pipeline.
    
    ignore_errors: boolean, if True, drops element that cause errors
    '''
    shuffle_buffer_size = int(shuffle_buffer_size)
    shuffle = shuffle_buffer_size > 0
    print('Loading dataset... Shuffle items? {}. Shuffle buffer: {}'.format(shuffle, shuffle_buffer_size))
    h5_paths = tf.data.Dataset.list_files(data_locations, shuffle=shuffle)
    
    filenames = tf.data.TFRecordDataset.list_files(data_locations)
    dataset = tf.data.TFRecordDataset(filenames)

    # Use `tf.parse_single_example()` to extract data from a `tf.Example`
    # protocol buffer, and perform any additional per-example processing.
    def parser(record):
        keys_to_features = {
            "path": tf.io.FixedLenFeature((), tf.string, ""),
            "sequence_index": tf.io.FixedLenFeature((), tf.int64, -1),
            "fft": tf.io.FixedLenFeature((), tf.string, ''),
            "fft_dimension_0": tf.io.FixedLenFeature((), tf.int64, -1),
            "fft_dimension_1": tf.io.FixedLenFeature((), tf.int64, -1),
            "image": tf.io.FixedLenFeature((), tf.string, ''),
            "image_dimension_0": tf.io.FixedLenFeature((), tf.int64, -1),
            "image_dimension_1": tf.io.FixedLenFeature((), tf.int64, -1),
        }
        parsed = tf.io.parse_single_example(record, keys_to_features)
        
        # Perform additional preprocessing on the parsed data.
        parsed['fft'] = tf.io.decode_raw(parsed['fft'], out_type=tf.float32)
        parsed['image'] = tf.io.decode_raw(parsed['image'], out_type=tf.float32)
        
        parsed['fft'] = tf.reshape(parsed['fft'], [parsed['fft_dimension_0'], parsed['fft_dimension_1'], 2])
        parsed['image'] = tf.reshape(parsed['image'], [parsed['image_dimension_0'], parsed['image_dimension_1'], 1])
        
        if preprocessing_function is not None:
            parsed = preprocessing_function(parsed)
        
        if include_all_parsed_features:
            return parsed
        
        # We only want input and expected output during training stage (X, Y)
        return parsed['fft'], parsed['image']
    
    # Use `Dataset.map()` to build a pair of a feature dictionary and a label
    # tensor for each example.
    dataset = dataset.map(parser)
    
    if ignore_errors:
        dataset = dataset.apply(tf.data.experimental.ignore_errors())
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=int(shuffle_buffer_size))
    dataset = dataset.batch(batch_size)

    # Each element of `dataset` is tuple of (input, output) pairs or a dictionary of features
    # (in which each value is a batch of values for that feature), and a batch of labels.
    return dataset
