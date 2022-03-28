#Public API's
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from bdpy.mri.image import export_brain_image
import numpy as np
import nibabel
from sklearn.metrics import mean_squared_error, matthews_corrcoef
import matplotlib.pyplot as plt
from scipy import stats
import json
import os
# Custom Imports
from fmri_models import *
import losses
from helper_functions import load_data, get_test_correlations, remove_low_variance_voxels
from keras.losses import cosine_similarity

# The name of the model to load
#model_name = 'CapsNet_fmri'
model_name = 'encoder_mach7'

# Directory paths
base_dir = ''
matlab_file = base_dir + 'kamitani_data/fmri/Subject3.mat'
test_image_ids = base_dir + 'kamitani_data/images/image_test_id.csv'
train_image_ids = base_dir + 'kamitani_data/images/image_training_id.csv'
images_npz = base_dir + 'kamitani_data/images/images_112.npz'
template= base_dir + 'kamitani_data/func_raw/sub-03_ses-perceptionTest01_task-perception_run-01_bold_preproc.nii.gz' # A template nifti file

# Load Data
print('Loading Data...')
# correlations = get_test_correlations(matlab_file, test_image_ids, train_image_ids, images_npz, roi='ROI_VC')
# print(correlations)
# exit()
x_train, x_test, y_train, y_test, xyz = load_data(matlab_file, test_image_ids, train_image_ids, images_npz, roi='ROI_VC')
NUM_VOXELS = y_train.shape[1]
print(NUM_VOXELS)
# y_test = y_test * 0.5

with open('models/' + model_name + '/train-history.json', 'r') as file:
    train_hist = json.load(file)

plt.plot(train_hist['loss'])
plt.plot(train_hist['val_loss'])
plt.show()
exit()

# Load model
# print('Loading Model...')
model = CapsEncoder(num_voxels=NUM_VOXELS)
print(model.class_name)
model.load_weights('models/' + model_name + '/model_weights').expect_partial()

# Calculating model predictions
print('Getting Model Predictions...')
y_pred_gaziv = np.loadtxt('models/gaziv_base_y_pred.csv', delimiter=',')
y_pred = model.predict(x_test)
np.savetxt('models/' + model_name + '/y_pred.csv', y_pred, delimiter=',')
# y_pred = np.loadtxt('models/' + model_name + '/y_pred.csv', delimiter=',')

print('Finishing Up...')
correlation = []
g_correlation = []
voxel_corr = []
vox_range = []
for i in range(NUM_VOXELS):
    voxel_corr.append(stats.pearsonr(y_pred[:, i], y_test[:, i])[0])
    vox_range.append(np.max(y_pred[:, i]) - np.min(y_pred[:, i]))
for i in range(50):
    correlation.append(stats.pearsonr(y_test[i], y_pred[i])[0]) # returns correlation coefficient r and two tailed p-value
    g_correlation.append(stats.pearsonr(y_test[i], y_pred_gaziv[i])[0])

# print(tf.shape(correlation))
# print(['{0:0.3f}'.format(i) for i in correlation])
print('Mean Sample Correlation: ', np.mean(correlation))

print('Mean Voxel Correlation: ', np.mean(voxel_corr))

# print(np.mean(cosine_similarity(y_test, y_pred)))

# plt.hist(correlation)
# plt.show()
# print(np.min(voxel_corr))

# print(correlation)
# #ind=36
# ind = np.argmax(correlation)
# print(correlation[ind])
# print(ind)
# plt.scatter(y_pred[ind], y_test[ind], s=0.5)
# plt.plot(y_test[ind], y_test[ind])
# plt.xlabel('Predicted Magnitude')
# plt.ylabel('GroundTruth Magnitude')
# plt.show()

# print(correlation)
# barloc = np.arange(len(correlation))
# width = 0.35
# fig, ax = plt.subplots()
# ax.bar(barloc, g_correlation, width=width, label='Gaziv Encoder')
# ax.bar(barloc+width, correlation, width=width, label='Capsule Encoder')
# ax.set_xlabel('Test Image Samples')
# ax.set_ylabel('Pearson Correlation Coefficient')
# ax.legend()
# plt.savefig('models/' + model_name + '/correlations.png')
# plt.show()

# slopes = []
# biases = []
# for i in range(50):
#     m, b = np.polyfit(y_pred[i], y_test[i], deg=1)
#     #m, b, _, _, _ = linregress(y_pred[i], Y_test_avg[i])
#     slopes.append(m)
#     biases.append(b)
# print('Slopes: ', slopes)
# print(np.mean(slopes))
# print('Biases: ', biases)
# print(np.mean(biases))


# ind = 36
# print(correlation[ind])
# plt.scatter(y_pred[ind], y_test[ind], s=0.5)
# plt.plot(y_test[ind], y_test[ind])
# plt.show()

# fig, axs = plt.subplots(5, 10)
# for ind in range(50):
#     axs[int(ind/10), ind%10].scatter(y_pred[ind], y_test[ind], s=0.5)
#     axs[int(ind/10), ind%10].plot(y_pred[ind], slopes[ind]*y_pred[ind] + biases[ind])
#     axs[int(ind/10), ind%10].plot(y_test[ind], y_test[ind])

# plt.show()

# y_mean = np.mean(y_train, axis=0)
# y_diff_s0 = abs(y_pred[4] - y_mean)
# y_diffs = [abs(pred - y_pred[0]) for pred in y_pred]
# print(np.shape(y_diffs))
# plt.hist(np.array(y_diffs).flatten())
# plt.show()
# y_diff_gaziv_s0 = abs(y_pred_gaziv[4] - y_mean)

# voxel_corr = np.array(voxel_corr)
# nifti = export_brain_image(y_diff_gaziv_s0, template=template, xyz=xyz)
# if not os.path.exists('nifti/' + model_name + '/'):
#     os.makedirs('nifti/' + model_name + '/')
# nibabel.save(nifti, 'nifti/' + model_name + '/sub03_s4_diff_from_mean.nii.gz')
# nibabel.save(nifti, 'nifti/gaziv/sub03_s4_diff_from_mean.nii.gz')
# nibabel.save(nifti, 'nifti/' + 'sub03_s4.nii.gz')
