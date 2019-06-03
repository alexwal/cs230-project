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

# Preprocessing functions

class fastMRIPreprocessor(object):
    '''
    Create a function for preprocessing fastMRI `tf.Example`s parsed from a TFRecord dataset.
    
    subsampling_mask_function: a function that accepts a 3-dimensional `shape` and returns an appropriately
    sized mask based on the second dimension.
    '''
    
    def __init__(self, normalize, shape, use_tiled_reflections, subsampling_mask_function):
        self.normalize = normalize
        self.shape = shape
        self.use_tiled_reflections = use_tiled_reflections
        self.subsampling_mask_function = subsampling_mask_function
    
    def __call__(self, example):
        
        # 1. If you're going to normalize or subsample, right away is the best time to do it on the
        # raw sensor data before other processing.
        
        if self.normalize:
            # Using rugh estimates for mean and std of fastMRI FFT data
            example['fft'] = (example['fft'] - 1e-8) / 1e-4
        
        if self.subsampling_mask_function is not None:
            # Subsample by multiplying FFT by subsampling mask
            mask = self.subsampling_mask_function((*FASTMRI_ORIGINAL_FFT_SHAPE, 1))
            fft = example['fft']
            fft *= mask
            fft.set_shape((*FASTMRI_ORIGINAL_FFT_SHAPE, 2))
            example['fft'] = fft
            
        if (self.normalize or self.subsampling_mask_function is not None) and self.shape is None:
            # Recalculate target reconstruction image (if self.shape is not None, it will do this too, so don't repeat)
            fft = combine_two_channels_of_complex_tensor(fft)
            fft.set_shape(FASTMRI_ORIGINAL_FFT_SHAPE)
            example['image'] = tf.expand_dims(tf.abs(signal_processing.tf_ifft2d(fft)), -1)
            
        # 2. Next, we may desire to reshape the data to a certain shape `shape` for ML models.
    
        if self.shape is not None:
            example = reshape_fastmri_tf_example(example, self.shape)
            
        # 3. After this, we may do some data augmentation.
            
        if self.use_tiled_reflections:
        
            image = example['image']
            tiled_image = tile_four_reflections(image)
            
            # New shape is one fourth the area of `tiled_image.shape` (because `tiled_image` has four reflections).
            crop_shape = (tiled_image.shape[0] // 2, tiled_image.shape[1] // 2, 1)
            
            random_crop = tf.image.random_crop(tiled_image, size=crop_shape)
            example['image'] = random_crop

            fft = signal_processing.tf_fft2d(tf.squeeze(random_crop))

            example['fft'] = fft
            
        # 4. Now, split FFT back into 2 channels
        
        fft = example['fft']
        
        if len(fft.shape) < 3:
            fft = tf.expand_dims(fft, -1)
        
        if fft.shape[-1] < 2:
            fft = tf.concat([tf.math.real(fft), tf.math.imag(fft)], -1)
        
        example['fft'] = fft
        
        return example

def center_crop(data, shape):
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

def reshape_fastmri_tf_example(example, shape):
    '''
    Resize fastMRI dataset `tf.Example`s parsed from a TFRecord dataset to `shape`.
    Performs FFT on resized reconstruction images and overwrites values for
    'image' and 'reconstruction' keys in `example`.
    '''
    assert len(shape) == 2, 'Shape must have exactly two integer elements.'
    
    # Create complex-valued matrix from 2 real and imaginary channels
    fft = combine_two_channels_of_complex_tensor(example['fft'])
    
    # Compute reconstruction using ifft2d on kspace (did confirm that output matches example['image']).
    fft.set_shape(FASTMRI_ORIGINAL_FFT_SHAPE)
    reconstruction = signal_processing.tf_ifft2d(fft)
    
    # Center crop reconstruction so that FFT and image shapes match
    reconstruction = center_crop(reconstruction, FASTMRI_ORIGINAL_IMAGE_SHAPE)
    
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

def tile_four_reflections(image):
    top_left = image 
    top_right = tf.image.flip_up_down(image)
    bottom_left = tf.image.flip_left_right(image)
    bottom_right = tf.image.flip_left_right(top_right)
    
    top = tf.concat([top_left, top_right], axis=1)
    bottom = tf.concat([bottom_left, bottom_right], axis=1)
    
    tiled = tf.concat([top, bottom], axis=0)
    
    return tiled

def combine_two_channels_of_complex_tensor(x):
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
