# File for training capsule network based encoders on the MNIST-fMRI dataset

print('Importing Packages...', flush=True)
#Public API's
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import pathlib
capsnet_path = pathlib.Path(__file__).parent.resolve().parent.resolve()
print('Base Dir: ', capsnet_path)
import sys
sys.path.append(str(capsnet_path)) # Allows imports from capsnet folder
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import json
import yaml
from scipy.io import loadmat
from scipy import stats
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import config
from skimage.transform import rescale, resize
# Custom Imports
from pycaps.fmri_models import *
import pycaps.losses as losses
from training.generators import ShiftGen

###############################################################################
# Set File Path Parameters
###############################################################################

save = True # Whether or not to save the model
scale = False
#base_dir = '/cluster/projects/uludag/shawn/CapsNet/'
base_dir = config.project_dir
model_name = 'digits_encoder_112x112_test1' # Model name

###############################################################################
# Set Model Hyperparameters
###############################################################################
training_params = {} # Dictionary to keep track of params so they can be saved
training_params['batch_size'] = 8
training_params['val_batch_size'] = 20
training_params['epochs'] = 100
training_params['roi'] = 'ROI_VC'
training_params['image_size'] = (112, 112)
#training_params['pretrain_file'] = 'imagenet-caffe-ref.mat'

loss = losses.mse_cosine_loss
training_params['loss'] = 'mse_cosine'
# loss = keras.losses.MeanSquaredError()
#training_params['loss'] = loss.get_config()
training_params['loss'] = 'mse_cosine'

optimizer = keras.optimizers.SGD(learning_rate=0.1, momentum=0.90, nesterov=True) # learning rate might be way to big
#optimizer = keras.optimizers.Adam(learning_rate=0.001)
training_params['optimizer'] = optimizer.get_config()

###############################################################################
# Load Data
###############################################################################
print('Loading Data...', flush=True)
dataset = loadmat(os.path.join(base_dir, 'handwritten_digits_in_fmri_dataset/69dataset_split.mat'))
x_test = dataset['x_test']
y_test = dataset['y_test']
x_train = dataset['x_train']
y_train = dataset['y_train']
x = dataset['x_all']
y = dataset['y_all']

# #scale data
scaler = StandardScaler()
y_train = scaler.fit_transform(y_train)
y_test  = scaler.transform(y_test)
y = scaler.fit_transform(y)

# Change Image resolution for experiments with high resolution mnist images
if training_params['image_size'] != (28, 28):
    print('Resizing Images...')
    temp = []
    for image in x_test:
        temp.append(resize(image[:, :, 0], output_shape=training_params['image_size'], order=3))
    x_test = np.expand_dims(np.array(temp), axis=-1)
    temp = []
    for image in x_train:
        temp.append(resize(image[:, :, 0], output_shape=training_params['image_size'], order=3))
    x_train = np.expand_dims(np.array(temp), axis=-1)

NUM_VOXELS=y_train.shape[1]

print(tf.config.list_physical_devices())

###############################################################################
# Train Model
###############################################################################
training_params['lr_schedule'] = True
if training_params['lr_schedule']:
    training_params['lr_schedule'] = {
        'gamma':0.1,
        'milestones': [40]}

def step_decay(epoch):
    lrate =training_params['optimizer']['learning_rate']
    for milestone in training_params['lr_schedule']['milestones']:
        if epoch >= milestone:
            lrate = training_params['lr_schedule']['gamma']*lrate
    return lrate

if training_params['lr_schedule']:
    callbacks = [keras.callbacks.LearningRateScheduler(step_decay, verbose=0)]
else:
    callbacks=[]

if save:
    if not os.path.exists(os.path.join('models', model_name)):
        os.makedirs(os.path.join('models', model_name))
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join('models', model_name, 'best_ckpt'),
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    callbacks.append(model_checkpoint_callback)

print('Starting Model Training', flush=True)

training_params['max_shift'] = 5
print(x_train.shape)
train_gen = ShiftGen(x_train, y_train, batch_size=training_params['batch_size'], max_shift=training_params['max_shift'])
model = CapsEncoder(num_voxels=NUM_VOXELS, num_output_capsules=2, routing='dynamic', caps_act='squash')

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=['mse', 'cosine_similarity', 'mae'],
)
model.build(input_shape=x_train.shape)
print(model.summary())

training = model.fit(train_gen, 
    batch_size=training_params['batch_size'], 
    epochs=training_params['epochs'],
    validation_data=(x_test, y_test),
    callbacks=callbacks,
    verbose=2)

###############################################################################
# Test and save model
###############################################################################
testing = model.evaluate(x_test, y_test, batch_size=training_params['val_batch_size'], return_dict=True)

# Model Saving
if save:
    # Save full model. Also save weights
    model.save_weights(os.path.join(capsnet_path, 'trained_models', model_name, 'model_weights'), save_format='tf')
    plt.plot(training.history['loss'])
    plt.plot(training.history['val_loss'])
    plt.legend(['train_loss', 'val_loss'])
    plt.savefig(os.path.join(capsnet_path, 'trained_models', model_name, '/val_loss.png'))

    with open(os.path.join(capsnet_path, 'trained_models', model_name, 'train_config.yaml'), 'w') as file:
        yaml.dump(training_params, file) # Save archetecture

    with open(os.path.join(capsnet_path, 'trained_models', model_name, 'config.yaml'), 'w') as file:
        yaml.dump(model.get_config(), file) # Save archetecture

    with open(os.path.join(capsnet_path, 'trained_models', model_name, 'test-history.json'), 'w') as file:
        json.dump(testing, file) # save testing history
        
    # convert lr to float64
    if 'lr' in training.history.keys():
        training.history['lr'] = [np.float64(lr) for lr in training.history['lr']]

    with open(os.path.join(capsnet_path, 'trained_models', model_name, '/train-history.json'), 'w') as file:
        json.dump(training.history, file) # save training history

