#Public API's
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
import numpy as np
# Custom Imports
from fmri_models import CapsEncoder, MatrixCapsEncoder
import losses
import json
from kamitani_data_handler import kamitani_data_handler as data_handler

save = True # Whether or not to save the model
#base_dir = '/cluster/projects/uludag/shawn/CapsNet/'
base_dir = ''
model_name = 'MatrixCapsNet_fmri' # Model name
matlab_file = base_dir + 'kamitani_data/fmri/Subject3.mat'
test_image_ids = base_dir + 'kamitani_data/images/image_test_id.csv'
train_image_ids = base_dir + 'kamitani_data/images/image_training_id.csv'
images_npz = base_dir + 'kamitani_data/images/images_112.npz'

#####################
# Load Data
#####################

# Note that the preprocessed data is in a wierd format, and it was easiest
# just to borrow the data handler from Gaziv et al.
data_handler = data_handler(matlab_file=matlab_file, test_img_csv=test_image_ids, train_img_csv=train_image_ids)

# Get fmri data
train_fmri, test_fmri, test_fmri_avg = data_handler.get_data(roi='ROI_VC')
NUM_VOXELS = train_fmri.shape[1]
'''
Note that each image is presented multiple times. Therefore there are more 
samples than there are images for both the training and test image sets

train_fmri: The preprocessed fmri data for the training images. 
test_fmri: The preprocessing fmri data for testing images
test_fmri_avg: The preprocessed fmri data of the test set averaged across
    all trials for each test image. (Ie the average activations for an image
    after multiple trials)

    shape: [samples, voxels]
'''

# Get images
npz_file = np.load(images_npz)
train_images = npz_file['train_images']
test_images = npz_file['test_images']
print(test_images.shape)

# Sort images so that they are in same order as fmri data
train_labels, test_labels = data_handler.get_labels()
# labels for the fmri data are the index of the corresponding image within the image_id csv's.
# since train_images and test_images are numpy arrays we can reorder them with a list of indices
# Note that test_labels is for test_fmri not test_fmri_avg
x_train = train_images[train_labels]
x_test = test_images # test_fmri_avg is already in order, hence do not need to reorder x_test

# rename fmri data
y_train = train_fmri
y_test = test_fmri_avg # For the test set, we only want one sample per test image, hence we use fmri averages across runs

###################
# Create Model
###################

# Create distributed learning strategy object
strategy = tf.distribute.MirroredStrategy()
print("Number of GPU's in use: {}".format(strategy.num_replicas_in_sync))

with strategy.scope():
    model = MatrixCapsEncoder(num_voxels=NUM_VOXELS)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=losses.mse_cosine_loss,
        metrics=['mse', 'cosine_similarity', 'mae']
    ) # No metric since we are only doing reconstructions loss
    
    model.build(x_train.shape)

# Print model summary
print(model.summary())

# Train and Test Model
training = model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), batch_size=3, epochs=10, verbose=1)
#testing = model.evaluate(x_test, y_test, batch_size=25, return_dict=True)

# Model Saving
if save:
    # Save model and model history
    model._set_inputs(x_train)
    model.save('models/' + model_name + '/saved_model', save_format='tf') # Saves whole model

    with open('models/' + model_name + '/train-history.json', 'w') as file:
        json.dump(training.history, file) # save training history

    with open('models/' + model_name + '/test-history.json', 'w') as file:
        json.dump(testing, file) # save testing history