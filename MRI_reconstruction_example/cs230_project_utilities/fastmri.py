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

from . import signal_processing


# Constants

FASTMRI_ORIGINAL_FFT_SHAPE = (640, 368)
FASTMRI_ORIGINAL_IMAGE_SHAPE = (320, 320)
FASTMRI_MODEL_INPUT_OUTPUT_SHAPE = (128, 128)

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

# Reading the raw dataset

def read_h5_file(path, coils, keep_original_reconstruction):
    '''
    Reads a fastMRI .h5 file from `path`, unwrapping data stored in the known
    keys.
    
    If `keep_original_reconstruction == False`, we will discard the original reconstruction images
    which are in key "reconstruction_esc" or "reconstruction_rss" and store `None`
    in 'original_reconstruction' key of returned dictionary.
    '''
    assert coils in ('single', 'multi'), 'Coils must be either \'single\' or \'multi\'.'
    
    with h5py.File(path, 'r') as f:
        metadata = f['ismrmrd_header'].value
        kspace = f['kspace'].value
        
        if keep_original_reconstruction:
            reconstruction_key = 'reconstruction_esc' if coils == 'single' else 'reconstruction_rss'
            original_reconstruction = f.get(reconstruction_key, default=None)
        
            if original_reconstruction is not None:
                original_reconstruction = original_reconstruction.value
        else:
            # If `keep_original_reconstruction == False`, store `None` in 'original_reconstruction' key.
            original_reconstruction = None
    
    return {'path': path,
            'coils': coils,
            'metadata': metadata,
            'kspace': kspace,
            'original_reconstruction': original_reconstruction}

# Convert to TFRecords

def _preprocess_fft_before_converting_to_tfrecords(fft, perform_subsampling):
    '''
    Resize fastMRI dataset `tf.Example`s parsed from a TFRecord dataset to `shape`.
    Performs FFT on resized reconstruction images and overwrites values for
    'image' key in `example`.
    
    Note: currently uses `example['fft']` even if `example['image']` present (which could be
    faster as it demands one less FFT).
    
    '''

    fft = _normalize(fft)
    fft, image =  _center_crop_and_reconstruct_fastmri_np_array(fft, FASTMRI_MODEL_INPUT_OUTPUT_SHAPE)

    if perform_subsampling:
        subsampling_mask_function =  SubsamplingMaskCreator(center_fractions=[0.08, 0.04], accelerations=[4, 8])
        mask = subsampling_mask_function((*fft.shape[:2], 1))
        fft *= mask

    return fft.numpy(), image.numpy()

def _tf_example_for_slice_in_mri_volume(index, h5_data, keep_original_reconstruction, perform_subsampling):
    
    # Unpack slice of data from h5 file
    path = h5_data['path']
    fft = h5_data['kspace'][index]
    if keep_original_reconstruction:
        original_image = h5_data['original_reconstruction'][index]
        assert original_image is not None, 'Error: `keep_original_reconstruction == True` but `original_image` is `None`.'
    else:
        original_image = None
    
    assert len(fft.shape) > 2 and fft.shape[-1] == 2, 'FFT must have 2 channels representing real and imaginary values.'
    
    fft, image = _preprocess_fft_before_converting_to_tfrecords(fft, perform_subsampling)
    
    if keep_original_reconstruction:
                               
        example = tf.train.Example(features=tf.train.Features(feature={
            'path': bytes_feature(path),
            'sequence_index': int64_feature(index),
            'fft': bytes_feature(fft.tostring()),
            # Save dimension so we can restore the key later.
            'fft_dimension_0': int64_feature(fft.shape[0]),
            'fft_dimension_1': int64_feature(fft.shape[1]),
            'image': bytes_feature(image.tostring()),
            'image_dimension_0': int64_feature(image.shape[0]),
            'image_dimension_1': int64_feature(image.shape[1]),
            'original_image': bytes_feature(original_image.tostring()),
            'original_image_dimension_0': int64_feature(original_image.shape[0]),
            'original_image_dimension_1': int64_feature(original_image.shape[1])
        }))
        
    else:
        
        # Don't include original reconstruction image
        
        # Create list of bytes with .tostring() for multidimensional ndarrays.
        example = tf.train.Example(features=tf.train.Features(feature={
            'path': bytes_feature(path),
            'sequence_index': int64_feature(index),
            'fft': bytes_feature(fft.tostring()),
            'fft_dimension_0': int64_feature(fft.shape[0]),
            'fft_dimension_1': int64_feature(fft.shape[1]),
            'image': bytes_feature(image.tostring()),
            'image_dimension_0': int64_feature(image.shape[0]),
            'image_dimension_1': int64_feature(image.shape[1]),
        }))
    

    
    return example

def _tf_examples_stored_in_h5_file(path, coils, keep_original_reconstruction, perform_subsampling):

    h5_data = read_h5_file(path, coils, keep_original_reconstruction)
    
    # Split complex kspace data into two channels (real and imaginary).
    kspace_sequence = h5_data['kspace']
    real_sequence = kspace_sequence.real
    imaginary_sequence = kspace_sequence.imag
    kspace_sequence = np.concatenate((np.expand_dims(real_sequence, axis=3),
                                   np.expand_dims(imaginary_sequence, axis=3)),
                                  axis=3)
    h5_data['kspace'] = kspace_sequence
    
    if keep_original_reconstruction:
        original_image_sequence = h5_data['original_reconstruction']
        if original_image_sequence is not None and len(original_image_sequence.shape) < 4:
            original_image_sequence = np.expand_dims(original_image_sequence, axis=-1) # model expects 4 dims

    examples = []
    for i in range(len(kspace_sequence)):
        example = _tf_example_for_slice_in_mri_volume(i, h5_data, keep_original_reconstruction, perform_subsampling)
        examples.append(example)
    
    return examples

def convert_fastmri_dataset_to_tfrecord_files(raw_data_locations, tfrecord_directory, coils,
                                              keep_original_reconstruction, perform_subsampling):
    tfrecord_file_pattern = os.path.join(tfrecord_directory, 'shard-{}.tfrecord')
    tfrecord_index = 0
    max_examples_per_tfrecord_file = 512
    examples_in_current_tfrecord_file = 0

    os.makedirs(os.path.dirname(tfrecord_file_pattern), exist_ok=True)
    
    tfrecord_path = tfrecord_file_pattern.format(tfrecord_index)
    writer = tf.io.TFRecordWriter(tfrecord_path)
    
    h5_paths = tf.data.Dataset.list_files(raw_data_locations, shuffle=True)
    
    for i, path in enumerate(h5_paths):
        path = path.numpy()
        print('Converting file {} at {} to TFRecords...'.format(i, path))
        try:
            examples = _tf_examples_stored_in_h5_file(path, coils, keep_original_reconstruction, perform_subsampling)
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
            raise

# Load TFRecords

def load_dataset(data_locations, batch_size, shuffle_buffer_size, load_original_reconstruction,
                 include_all_parsed_features, ignore_errors, perform_data_augmentation):
    '''
    Returns iterator of fastMRI data located in `data_locations`.
    
    data_locations:  A string, a list of strings, or a `tf.Tensor` of string type
    (scalar or vector), representing the filename glob (i.e. shell wildcard)
    pattern(s) that will be matched.
    
    preprocessing_function: A function that takes in a parsed example and prepares it
    for use in a training or evaluation tf.data.Dataset pipeline.
    
    ignore_errors: boolean, if True, drops element that cause errors
    
    load_reconstruction: boolean, whether to read 'image' key from `tf.Example`s on disk.
    
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
        if load_original_reconstruction:
            keys_to_features = {
                "path": tf.io.FixedLenFeature((), tf.string, ''),
                "sequence_index": tf.io.FixedLenFeature((), tf.int64, -1),
                "fft": tf.io.FixedLenFeature((), tf.string, ''),
                "fft_dimension_0": tf.io.FixedLenFeature((), tf.int64, -1),
                "fft_dimension_1": tf.io.FixedLenFeature((), tf.int64, -1),
                "image": tf.io.FixedLenFeature((), tf.string, ''),
                "image_dimension_0": tf.io.FixedLenFeature((), tf.int64, -1),
                "image_dimension_1": tf.io.FixedLenFeature((), tf.int64, -1),
                "original_image": tf.io.FixedLenFeature((), tf.string, ''),
                "original_image_dimension_0": tf.io.FixedLenFeature((), tf.int64, -1),
                "original_image_dimension_1": tf.io.FixedLenFeature((), tf.int64, -1)
            }
        else:
            keys_to_features = {
                "path": tf.io.FixedLenFeature((), tf.string, ''),
                "sequence_index": tf.io.FixedLenFeature((), tf.int64, -1),
                "fft": tf.io.FixedLenFeature((), tf.string, ''),
                "fft_dimension_0": tf.io.FixedLenFeature((), tf.int64, -1),
                "fft_dimension_1": tf.io.FixedLenFeature((), tf.int64, -1),
                "image": tf.io.FixedLenFeature((), tf.string, ''),
                "image_dimension_0": tf.io.FixedLenFeature((), tf.int64, -1),
                "image_dimension_1": tf.io.FixedLenFeature((), tf.int64, -1)
            }

        parsed = tf.io.parse_single_example(record, keys_to_features)
        
        # Perform additional preprocessing on the parsed data.
        parsed['fft'] = tf.io.decode_raw(parsed['fft'], out_type=tf.float32)
        parsed['fft'] = tf.reshape(parsed['fft'], [parsed['fft_dimension_0'], parsed['fft_dimension_1'], 2])
        
        parsed['image'] = tf.io.decode_raw(parsed['image'], out_type=tf.float32)
        parsed['image'] = tf.reshape(parsed['image'], [parsed['image_dimension_0'], parsed['image_dimension_1'], 1])
        
        if load_original_reconstruction:
            parsed['orignal_image'] = tf.io.decode_raw(parsed['orignal_image'], out_type=tf.float32)
            parsed['orignal_image'] = tf.reshape(parsed['orignal_image'], [parsed['original_image_dimension_0'], parsed['original_image_dimension_1'], 1])
            
        else:
            # Set 'orignal_image' key to `None` just to be safe.
            parsed['orignal_image'] = None
        
        if perform_data_augmentation:
            _augment_with_tiled_reflections_and_random_crop(parsed)
        
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

# Preprocessing functions
    
def _augment_with_tiled_reflections_and_random_crop(example):
    image = example['image']
    image.set_shape((*FASTMRI_MODEL_INPUT_OUTPUT_SHAPE, 1)) # needs shape for tiling
    tiled_image = _tile_four_reflections(image)
    # New shape is one fourth the area of `tiled_image.shape` (because `tiled_image` has four reflections).
    crop_shape = (tiled_image.shape[0] // 2, tiled_image.shape[1] // 2, 1)
    random_crop = tf.image.random_crop(tiled_image, size=crop_shape)
    example['image'] = random_crop

    fft = signal_processing.tf_fft2d(tf.squeeze(random_crop))
    fft = tf.expand_dims(fft, -1)
    fft = tf.concat([tf.math.real(fft), tf.math.imag(fft)], -1)
    example['fft'] = fft
    
def _normalize(fft):
    fft = (fft - 1e-8) / 1e-4
    return fft

def _center_crop(data, shape):
    """
    Apply a center crop to the input image or batch of complex images.
    Args:
        data (np.ndarray): The complex input tensor to be center cropped. It should
            have at least 2 dimensions and the cropping is applied along dimensions
            0 and 1.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.
    Returns:
        np.ndarray: The center cropped image
    """
    assert 0 < shape[0] <= data.shape[0]
    assert 0 < shape[1] <= data.shape[1]
    w_from = (data.shape[0] - shape[0]) // 2
    h_from = (data.shape[1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[w_from:w_to, h_from:h_to, ...]

def _center_crop_and_reconstruct_fastmri_np_array(fft, shape):
    '''
    Resize fastMRI dataset `tf.Example`s parsed from a TFRecord dataset to `shape`.
    Performs FFT on resized reconstruction images and overwrites values for
    'image' key in `example`.
    
    Note: currently uses `example['fft']` even if `example['image']` present (which could be
    faster as it demands one less FFT).
    
    '''
    assert len(shape) == 2, 'Shape must have exactly two integer elements.'
    
    # Create complex-valued matrix from 2 real and imaginary channels
    fft = _combine_two_channels_of_complex_tensor(fft)
    
    # Compute reconstruction using ifft2d on kspace
    reconstruction = signal_processing.tf_ifft2d(fft)
    
    # Center crop reconstruction so that FFT and image shapes match
    reconstruction = _center_crop(reconstruction, FASTMRI_ORIGINAL_IMAGE_SHAPE)
    
    # Resize reconstruction image to desired shape
    reconstruction = tf.expand_dims(reconstruction, -1)
    reconstruction_real = tf.image.resize(tf.math.real(reconstruction),
                                                     size=shape,
                                                     method='lanczos3',
                                                     preserve_aspect_ratio=True,
                                                     antialias=False)
    reconstruction_imaginary = tf.image.resize(tf.math.imag(reconstruction),
                                                     size=shape,
                                                     method='lanczos3',
                                                     preserve_aspect_ratio=True,
                                                     antialias=False)
    
    reconstruction = tf.complex(real=reconstruction_real, imag=reconstruction_imaginary)

    # Add key, value for resized reconstruction
    image = tf.abs(reconstruction)
    
    # Finally, recompute and store FFT (kspace data) based on resized reconstruction image
    fft = signal_processing.tf_fft2d(tf.squeeze(reconstruction))
    fft = tf.expand_dims(fft, -1)
    fft = tf.concat([tf.math.real(fft), tf.math.imag(fft)], -1)

    return fft, image

def _center_crop_fastmri_tf_example(example, shape):
    '''
    Resize fastMRI dataset `tf.Example`s parsed from a TFRecord dataset to `shape`.
    Performs FFT on resized reconstruction images and overwrites values for
    'image' key in `example`.
    
    Note: currently uses `example['fft']` even if `example['image']` present (which could be
    faster as it demands one less FFT).
    
    '''
    assert len(shape) == 2, 'Shape must have exactly two integer elements.'
    
    # Create complex-valued matrix from 2 real and imaginary channels
    fft = combine_two_channels_of_complex_tensor(example['fft'])
    
    # Compute reconstruction using ifft2d on kspace
    fft.set_shape(FASTMRI_ORIGINAL_FFT_SHAPE)
    reconstruction = signal_processing.tf_ifft2d(fft)
    
    # Center crop reconstruction so that FFT and image shapes match
    reconstruction = _center_crop(reconstruction, FASTMRI_ORIGINAL_IMAGE_SHAPE)
    
    # Resize reconstruction image to desired shape
    reconstruction = tf.expand_dims(reconstruction, -1)
    reconstruction_real = tf.image.resize(tf.math.real(reconstruction),
                                                     size=shape,
                                                     method='lanczos3',
                                                     preserve_aspect_ratio=True,
                                                     antialias=False)
    reconstruction_imaginary = tf.image.resize(tf.math.imag(reconstruction),
                                                     size=shape,
                                                     method='lanczos3',
                                                     preserve_aspect_ratio=True,
                                                     antialias=False)
    
    reconstruction = tf.complex(real=reconstruction_real, imag=reconstruction_imaginary)

    # Add key, value for resized reconstruction
    example['image'] = tf.abs(reconstruction)
    
    # Finally, recompute and store FFT (kspace data) based on resized reconstruction image
    fft = signal_processing.tf_fft2d(tf.squeeze(reconstruction))
    fft = tf.expand_dims(fft, -1)
    fft = tf.concat([tf.math.real(fft), tf.math.imag(fft)], -1)
    example['fft'] = fft

    return example

def _tile_four_reflections(image):
    top_left = image 
    top_right = tf.image.flip_up_down(image)
    bottom_left = tf.image.flip_left_right(image)
    bottom_right = tf.image.flip_left_right(top_right)
    
    top = tf.concat([top_left, top_right], axis=1)
    bottom = tf.concat([bottom_left, bottom_right], axis=1)
    
    tiled = tf.concat([top, bottom], axis=0)
    
    return tiled

def _combine_two_channels_of_complex_tensor(x):
    # We store real and imag values of a complex number in channels 0, 1 when needed.
    assert x.shape[-1] == 2, 'Input must have exactly two channels (as final dimension).'
    x = tf.complex(real=x[..., 0], imag=x[..., 1])
    return x

# Subsampling mask (adapted from: https://github.com/facebookresearch/fastMRI)

class SubsamplingMaskCreator(object):
    """
    SubsamplingMask is used to instantiate a sub-sampling mask of a given shape.
    
    The mask selects a subset of columns from the input k-space data. If the k-space data has N
    columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center corresponding to
           low-frequencies
        2. The other columns are selected uniformly at random with a probability equal to:
           prob = (N / acceleration - N_low_freqs) / (N - N_low_freqs).
    This ensures that the expected number of columns selected is equal to (N / acceleration)
    It is possible to use multiple center_fractions and accelerations, in which case one possible
    (center_fraction, acceleration) is chosen uniformly at random each time the MaskFunc object is
    called.
    For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04], then there
    is a 50% probability that 4-fold acceleration with 8% center fraction is selected and a 50%
    probability that 8-fold acceleration with 4% center fraction is selected.
    """

    def __init__(self, center_fractions, accelerations):
        """
        Args:
            center_fractions (List[float]): Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is chosen uniformly
                each time.
            accelerations (List[int]): Amount of under-sampling. This should have the same length
                as center_fractions. If multiple values are provided, then one of these is chosen
                uniformly each time. An acceleration of 4 retains 25% of the columns, but they may
                not be spaced evenly.
        """
        if len(center_fractions) != len(accelerations):
            raise ValueError('Number of center fractions should match number of accelerations')

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.rng = np.random.RandomState()

    def __call__(self, shape, seed=None):
        """
        Args:
            shape (iterable[int]): The shape of the mask to be created. The shape should have
                at least 3 dimensions. Samples are drawn along the second last dimension.
            seed (int, optional): Seed for the random number generator. Setting the seed
                ensures the same mask is generated each time for the same shape.
        Returns:
            torch.Tensor: A mask of the specified shape.
        """
        if len(shape) < 3:
            raise ValueError('Shape should have 3 or more dimensions')

        self.rng.seed(seed)
        num_cols = shape[-2]

        choice = self.rng.randint(0, len(self.accelerations))
        center_fraction = self.center_fractions[choice]
        acceleration = self.accelerations[choice]

        # Create the mask
        num_low_freqs = int(round(num_cols * center_fraction))

        prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)
        mask = self.rng.uniform(size=num_cols) < prob
        
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad:pad + num_low_freqs] = True

        # Reshape the mask
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols
        
        mask = np.reshape(mask, mask_shape).astype(np.float32)

        return mask
