# Public API's
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
import bdpy
from bdpy.mri.image import export_brain_image
import numpy as np
import nibabel
from sklearn.metrics import mean_squared_error
# Custom Imports
from fmri_models import CapsEncoder
import losses
from kamitani_data_handler import kamitani_data_handler as data_handler

# The name of the model to load
model_name = 'CapsNet_fmri'

# Directory paths
base_dir = ''
matlab_file = base_dir + 'kamitani_data/fmri/Subject3.mat'
test_image_ids = base_dir + 'kamitani_data/images/image_test_id.csv'
train_image_ids = base_dir + 'kamitani_data/images/image_training_id.csv'
images_npz = base_dir + 'kamitani_data/images/images_112.npz'
template= base_dir + 'kamitani_data/func_raw/sub-03_ses-perceptionTest01_task-perception_run-01_bold_preproc.nii.gz' # A template nifti file

# Load Model
model = keras.models.load_model('models/' + model_name + '/saved_model', custom_objects={'mse_recon_loss' : mean_squared_error})

# Load Data
data_handler = data_handler(matlab_file=matlab_file, test_img_csv=test_image_ids, train_img_csv=train_image_ids)
train_fmri, test_fmri, test_fmri_avg = data_handler.get_data(roi='ROI_VC')
NUM_VOXELS = train_fmri.shape[1]

npz_file = np.load(images_npz)
test_images = npz_file['test_images']
print('Test Images Shape: ', test_images.shape)

train_labels, test_labels = data_handler.get_labels()
x_test = test_images # test_fmri_avg is already in order, hence do not need to reorder x_test
y_test = test_fmri_avg # For the test set, we only want one sample per test image, hence we use fmri averages across runs

# Get one reconstruction
sample = tf.expand_dims(x_test[0], axis=0) #(1, NUM_VOXELS)
recon_fmri = model.call(sample) # (1, NUM_VOXELS)
ae = np.abs(y_test[0] - recon_fmri[0])

# Get voxel coordinates
xyz, _ = data_handler.get_voxel_loc()

# Convert fmri data to nifti images
# nifti_original = export_brain_image(y_test[0], template=template, xyz=xyz)
# nifti_recon = export_brain_image(recon_fmri, template=template, xyz=xyz)
nifti_ae = export_brain_image(ae, template=template, xyz=xyz)

# # save fmri images
# nibabel.save(nifti_original, 'nifti/sub03_sample0.nii.gz')
# nibabel.save(nifti_recon, 'nifti/sub03_sample0_recon.nii.gz')
nibabel.save(nifti_ae, 'nifti/sub03_caps-ae_s0.nii.gz')