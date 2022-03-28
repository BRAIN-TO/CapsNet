#Public API's
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from bdpy.mri.image import export_brain_image
import numpy as np
import nibabel
from sklearn.metrics import mean_squared_error
# Custom Imports
from fmri_models import *
import losses
from helper_functions import load_data

# The name of the model to load
#model_name = 'CapsNet_fmri'
model_name = 'encoder_mach6_msle'

# Directory paths
base_dir = ''
matlab_file = base_dir + 'kamitani_data/fmri/Subject3.mat'
test_image_ids = base_dir + 'kamitani_data/images/image_test_id.csv'
train_image_ids = base_dir + 'kamitani_data/images/image_training_id.csv'
images_npz = base_dir + 'kamitani_data/images/images_112.npz'
template= base_dir + 'kamitani_data/func_raw/sub-03_ses-perceptionTest01_task-perception_run-01_bold_preproc.nii.gz' # A template nifti file

# Load Data
print('Loading Data...')
x_train, x_test, y_train, y_test, xyz = load_data(matlab_file, test_image_ids, train_image_ids, images_npz, roi='ROI_VC')
NUM_VOXELS = y_train.shape[1]

# Load model
print('Loading Model...')
model = EncoderMach6(num_voxels=NUM_VOXELS)
#model = CapsEncoder(num_voxels=NUM_VOXELS)
model.load_weights('models/' + model_name + '/model_weights').expect_partial()

print('Calculating Model Predictions...')
sample = tf.expand_dims(x_test[0], axis=0, name='inputs') # sample: (1, im_h, im_w, num_chan)
recon_fmri = model.predict(sample) # recon_fmri: (1, NUM_VOXELS)
print(recon_fmri)
convlayer = model.get_layer(index=2)
print(convlayer.get_config())
weights = convlayer.get_weights()
print(len(weights))
print(np.shape(weights[0]))
print(np.shape(weights[1]))
exit()

#ae = np.abs(y_test[0] - recon_fmri[0])

# Convert fmri data to nifti images
print('Exporting Files...')
#nifti_original = export_brain_image(y_test[10], template=template, xyz=xyz)
# nifti_recon = export_brain_image(recon_fmri, template=template, xyz=xyz)
#nifti_ae = export_brain_image(ae, template=template, xyz=xyz)

# # save fmri images
#nibabel.save(nifti_original, 'nifti/sub03_sample10.nii.gz')
# nibabel.save(nifti_recon, 'nifti/' + model_name + '/sub03_s0_em2.nii.gz')
#nibabel.save(nifti_ae, 'nifti/' + model_name + '/sub03_caps-ae_s0.nii.gz')
