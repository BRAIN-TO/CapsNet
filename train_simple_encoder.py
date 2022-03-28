print('Importing Packages...', flush=True)
#Public API's
from gc import callbacks
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import json
import yaml
from scipy.io import loadmat
from scipy import stats
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import config
# Custom Imports
from fmri_models import *
import losses
from generators import ShiftGen
import tqdm

###############################################################################
# Set File Path Parameters
###############################################################################

save = True # Whether or not to save the model
scale = False
#base_dir = '/cluster/projects/uludag/shawn/CapsNet/'
base_dir = config.project_dir
model_name = 'digits_encoder_test4' # Model name

###############################################################################
# Set Model Hyperparameters
###############################################################################
training_params = {} # Dictionary to keep track of params so they can be saved
training_params['batch_size'] = 8
training_params['val_batch_size'] = 20
training_params['epochs'] = 1200
training_params['roi'] = 'ROI_VC'
#training_params['pretrain_file'] = 'imagenet-caffe-ref.mat'

loss = losses.mse_cosine_loss
training_params['loss'] = 'mse_cosine'
# loss = keras.losses.MeanSquaredError()
#training_params['loss'] = loss.get_config()
training_params['loss'] = 'mse_cosine'

#optimizer = keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True) # learning rate might be way to big
optimizer = keras.optimizers.Adam()
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

#scale data
scaler = StandardScaler()
y_train = scaler.fit_transform(y_train)
y_test  = scaler.transform(y_test)
y = scaler.fit_transform(y)

NUM_VOXELS=y_train.shape[1]
###############################################################################
# Create Model
###############################################################################

# Create distributed learning strategy object
# strategy = tf.distribute.MirroredStrategy()
# print("Number of devices in use: {}".format(strategy.num_replicas_in_sync))

# with strategy.scope():


###############################################################################
# Train Model
###############################################################################
def step_decay(epoch):
    lrate = 0.001
    if(epoch>10):
        lrate = 0.0001
    if (epoch > 20):
        lrate = 0.00001
    # if (epoch > 30):
    #     lrate = 0.000001
    # if (epoch > 200):
    #     lrate = 0.0000001
    return lrate

training_params['lr_schedule'] = 'e<10: 0.001, e>10: 0.0001, e>20: 0.00001'

lr_scheduler = keras.callbacks.LearningRateScheduler(step_decay)

print('Starting Model Training', flush=True)
# n_splits = 5
# kfold = KFold(n_splits=n_splits)
# fold_no = 1
# scores = []
# y_pred = []
# for train, test in tqdm.tqdm(kfold.split(x, y), total=n_splits):
#     train_gen = ShiftGen(x[train], y[train], batch_size=training_params['batch_size'], max_shift=5)
#     model = CapsEncoder(num_voxels=NUM_VOXELS)
#     model.compile(
#         optimizer=optimizer,
#         loss=loss,
#         metrics=['mse', 'cosine_similarity', 'mae', pearson],
#     )
#     training = model.fit(train_gen, 
#         batch_size=training_params['batch_size'], 
#         epochs=training_params['epochs'],
#         #callbacks=[lr_scheduler],
#         verbose=0)
#     fold_scores = model.evaluate(x[test], y[test], verbose=0)
#     scores.append(fold_scores)
#     fold_no = fold_no + 1
#     y_pred.append(model.predict(x[test]))

# scores = np.mean(scores, axis=0)
# print(model.metrics_names)
# print(scores)
# score_dict = [{metric : scores[i]} for metric, i in enumerate(model.metrics_names)]
# with open('models/digits_encoder/cv_scores.yaml', 'w') as file:
#     yaml.dump(score_dict, file)

training_params['max_shift'] = 5
train_gen = ShiftGen(x_train, y_train, batch_size=training_params['batch_size'], max_shift=training_params['max_shift'])
model = CapsEncoder(num_voxels=NUM_VOXELS)
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
    callbacks=[lr_scheduler],
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
        yaml.dump(training_params, file) # Save archetecture

    with open('models/' + model_name + '/config.yaml', 'w') as file:
        yaml.dump(model.get_config(), file) # Save archetecture

    with open('models/' + model_name + '/test-history.json', 'w') as file:
        # df = pd.DataFrame(testing)
        # df.to_json(file)
        json.dump(testing, file) # save testing history
        # convert lr to float64

    if 'lr' in training.history.keys():
        training.history['lr'] = [np.float64(lr) for lr in training.history['lr']]

    with open('models/' + model_name + '/train-history.json', 'w') as file:
        # df = pd.DataFrame(training.history)
        # df.to_json(file)
        json.dump(training.history, file) # save training history

