print('Importing Packages...', flush=True)
#Public API's
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import json
import yaml
import matplotlib.pyplot as plt
import time
# Custom Imports
from fmri_models import *
import losses
from kamitani_data_handler import kamitani_data_handler as data_handler
from helper_functions import load_data, remove_low_variance_voxels
from generators import ShiftGen

ts = time.time()

###############################################################################
# Set File Path Parameters
###############################################################################

save = True # Whether or not to save the model
scale = False
base_dir = '/cluster/projects/uludag/shawn/CapsNet/'
#base_dir = ''
model_name = 'capsencoder_sgd' # Model name
pretrain_weights = base_dir + 'imagenet-caffe-ref.mat'
matlab_file = base_dir + 'kamitani_data/fmri/Subject3.mat' # contains processed fmri data
test_image_ids = base_dir + 'kamitani_data/images/image_test_id.csv'
train_image_ids = base_dir + 'kamitani_data/images/image_training_id.csv'
images_npz = base_dir + 'kamitani_data/images/images_112.npz'

###############################################################################
# Set Model Hyperparameters
###############################################################################
training_params = {} # Dictionary to keep track of params so they can be saved
training_params['batch_size'] = 75
training_params['val_batch_size'] = 50
training_params['epochs'] = 80
training_params['roi'] = 'ROI_VC'
#training_params['pretrain_file'] = 'imagenet-caffe-ref.mat'

loss = losses.mse_cosine_loss
# loss = keras.losses.MeanSquaredError()
# training_params['loss'] = loss.get_config()
training_params['loss'] = 'mse_cosine'

optimizer = keras.optimizers.SGD(learning_rate=0.1, decay=0.1/training_params['epochs'], momentum=0.9, nesterov=True, clipvalue=0.5) # learning rate might be way to big
#optimizer = keras.optimizers.Adam()
training_params['optimizer'] = optimizer.get_config()

###############################################################################
# Load Data
###############################################################################
print('Loading Data...', flush=True)
x_train, x_test, y_train, y_test, xyz = load_data(matlab_file, test_image_ids, train_image_ids, images_npz, roi=training_params['roi'])

# y_train[y_train < 0.2] = 0
# y_test[y_test < 0.2] = 0

# training_params['bins'] = [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9]
# assert len(training_params['bins']) % 2 == 0, 'must have an even number of bins boundaries in order for symettry'

# # Quantize fmri data
# y_train = np.digitize(y_train, training_params['bins'])
# y_test = np.digitize(y_test, training_params['bins'])

# # Center indices from np.digitize around zero
# y_train = y_train - int(len(training_params['bins']) / 2)
# y_test = y_test - int(len(training_params['bins']) / 2)

# # change type back to float
# y_train = y_train.astype('float32')
# y_test = y_test.astype('float32')

# Center indices of bins to reflect data
NUM_VOXELS = y_train.shape[1]

# y_test, y_train, xyz = remove_low_variance_voxels(y_test, y_train, xyz, threshold=0.08)
# NUM_VOXELS = y_train.shape[1]

if scale is True:
    y_train = y_train * 0.50
    y_test = y_test * 0.50

###############################################################################
# Create Model
###############################################################################

# Create distributed learning strategy object
strategy = tf.distribute.MirroredStrategy()
print("Number of devices in use: {}".format(strategy.num_replicas_in_sync))

with strategy.scope():
    print('Compiling Model...', flush=True)
    model = CapsEncoder(num_voxels=NUM_VOXELS)
    print('model type: ', model.class_name)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['mse', 'cosine_similarity', 'mae'],
    )
    print('Building Model...', flush=True)
    model.build(x_train.shape)

# Print model summary
print(model.summary(), flush=True)

print('Starting Model Training', flush=True)

###############################################################################
# Train Model
###############################################################################
def step_decay(epoch):
    lrate = 0.1
    if(epoch>20):
        lrate = 0.01
    if (epoch > 35):
        lrate = 0.001
    if (epoch > 45):
        lrate = 0.0001
    if (epoch > 50):
        lrate = 0.00001
    return lrate

lr_scheduler = keras.callbacks.LearningRateScheduler(step_decay)

train_gen = ShiftGen(x_train, y_train, batch_size=training_params['batch_size'], max_shift=5)

training = model.fit(train_gen, 
    validation_data=(x_test, y_test), 
    batch_size=training_params['batch_size'], 
    epochs=training_params['epochs'],
    #callbacks=[lr_scheduler],
    verbose=2)

###############################################################################
# Test and save model
###############################################################################
testing = model.evaluate(x_test, y_test, batch_size=training_params['val_batch_size'], return_dict=True)

# Model Saving
if save:
    # Save full model. Also save weights
    model.save_weights('models/' + model_name + '/model_weights', save_format='tf')
    plt.plot(training.history['loss'])
    plt.plot(training.history['val_loss'])
    plt.legend(['train_loss', 'val_loss'])
    plt.savefig('models/' + model_name + '/val_loss.png')

    with open('models/' + model_name + '/train_config.yaml', 'w') as file:
        yaml.dump(training_params, file) # Save training hyperparams

    with open('models/' + model_name + '/config.yaml', 'w') as file:
        yaml.dump(model.get_config(), file) # Save archetecture

    with open('models/' + model_name + '/train-history.json', 'w') as file:
        json.dump(training.history, file) # save training history

    with open('models/' + model_name + '/test-history.json', 'w') as file:
        json.dump(testing, file) # save testing history

print('Total Runtime: ', time.time() - ts)