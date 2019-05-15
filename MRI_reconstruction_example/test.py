
# coding: utf-8

# In[1]:


# Autoreload changed python modules
#get_ipython().magic('load_ext autoreload')


# In[2]:


import os, glob
import h5py # for loading .h5 files
import numpy as np
# get_ipython().magic('matplotlib inline')


# In[3]:



# In[4]:


import cs230_project_utilities as utils
# make sure you have pywt: pip3 install PyWavelets --user


# # Loading the data

# In[5]:


# Location of directory H5Exports_AnimiX/ (downloaded from Olivier's link)
dataset_directory = '/scratch/groups/mwinterm/BoostMRI/data/H5Exports_AnimiX'


# In[6]:


# Find all the files in our dataset
h5_files = utils.automap.find_dataset_files(dataset_directory)


# # Visualizing the data

# In[7]:


##### Finally, we can see the raw data
h5 = h5_files['1497_18747']['h5']
h5_data = utils.automap.read_h5_file(h5)
print(h5_data.keys())

images = h5_data['images']
magnitude = h5_data['magnitude']
phase = h5_data['phase']
classification = h5_data['classification']

# (Note: shape of magnitude and phase are different from image)
print(images.shape, magnitude.shape, phase.shape)
print(np.ndarray.flatten(classification))


# In[8]:


sample_index = np.argmax(classification)


# In[9]:

# In[10]:


# Construct FFT (k-space data) from magnitude and phase
fft = magnitude[sample_index] * np.exp(1j * phase[sample_index])

# Take the inverse FFT
ifft = utils.signal_processing.ifft_2D_centered(fft)

# Note: shape of magnitude and phase are different from image.
# Because of this, the reconstruction shape is different from the
# image shape and so we can't compare the image and reconstruction directly.
# How will we solve this?

# This check to make sure we are correctly combining magnitude and phase to construct the full, complex-valued FFT.
print('Error in FFT magnitude: {}'.format(utils.signal_processing.mean_square_error(np.abs(fft), magnitude[sample_index])))
print('Error in FFT phase: {}'.format(utils.signal_processing.mean_square_error(np.angle(fft), phase[sample_index])))

# # Automap Model

# In[12]:


import keras
from keras.layers import Input,Conv2D, Conv2DTranspose, Dense, Reshape, Flatten
from keras.models import Model
from keras.optimizers import RMSprop
from keras import losses


# In[13]:


def load_model():
    n_H, n_W = 256, 240
    X = Input((n_H, n_W,2))
    X1 = Flatten()(X)
    fc1 = Dense(n_H * n_W * 2, activation = 'tanh')(X1)
    fc2 = Dense(n_H * n_W, activation = 'tanh')(fc1)
    fc3 = Dense(n_H * n_W, activation = 'tanh')(fc2)
    X2 = Reshape((n_H, n_W, 1))(fc3)
    conv1_1 = Conv2D(64, 5, activation='relu', padding='same')(X2)
    conv1_2 = Conv2D(64, 5, activation='relu', padding='same')(conv1_1)
    conv1_3 = Conv2DTranspose(64, 9, activation='relu', padding='same')(conv1_2)
    out = Conv2D(1, 1, activation = 'linear',padding='same')(conv1_3)
    model=Model(inputs=X, outputs=out)
    model.compile(optimizer=RMSprop(lr=1e-5), loss='mean_squared_error')
    return model


# In[14]:


model = load_model()


# In[15]:


train_dataset = {
    'input': {'fft': [], 'class': []}, # currently storing the sequences of ~15 images right now
    'output': []
}

num_sequences, max_sequences = 0, 10
for _, h5_file in h5_files.items():
    
    if num_sequences >= max_sequences:
        break
    
    h5_data = utils.automap.read_h5_file(h5_file['h5'])
    
    image_sequence = h5_data['images']
    image_sequence = utils.automap.transform_training_set_image_to_match_automap_output(image_sequence)
    image_sequence = np.transpose(image_sequence, axes=(2, 0, 1))
    
    magnitude_sequence = h5_data['magnitude']
    phase_sequence = h5_data['phase']
    fft_sequence = np.concatenate((np.expand_dims(magnitude_sequence, axis=3),
                          np.expand_dims(phase_sequence, axis=3)),
                         axis=3)
    
    class_sequence = h5_data['classification']

    train_dataset['input']['fft'].append(fft_sequence)     
    train_dataset['input']['class'].append(class_sequence)    
    train_dataset['output'].append(image_sequence)
    
    num_sequences += 1


# In[16]:


fft_sequence = train_dataset['input']['fft'][0]
image_sequence = train_dataset['output'][0]
image_sequence = np.expand_dims(image_sequence, axis=-1)
print(fft_sequence.shape, image_sequence.shape)


# In[19]:


fft_sequence = fft_sequence.astype(np.float64)


# In[ ]:


hist00 = model.fit(fft_sequence, image_sequence, batch_size=len(fft_sequence), epochs=3, verbose=1)

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





