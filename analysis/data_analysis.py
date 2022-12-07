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
from bdpy.mri.image import export_brain_image
import numpy as np
import nibabel
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt
from scipy import stats
import json
import os
import tensorflow_probability as tfp
# Custom Imports
from pycaps.fmri_models import *
from misc.helper_functions import *
from keras.losses import cosine_similarity, mean_squared_error, mean_absolute_error
from misc.kamitani_data_handler import kamitani_data_handler

# The name of the model to load
#model_name = 'CapsNet_fmri'
model_name = 'caps_encoder_test2'

# Directory paths
base_dir = str(capsnet_path)
matlab_file = os.path.join(base_dir, 'kamitani_data/fmri/Subject3.mat')
test_image_ids = os.path.join(base_dir, 'kamitani_data/images/image_test_id.csv')
train_image_ids = os.path.join(base_dir, 'kamitani_data/images/image_training_id.csv')
images_npz = os.path.join(base_dir, 'kamitani_data/images/images_112.npz')
template= os.path.join(base_dir, 'kamitani_data/func_raw/sub-03_ses-perceptionTest01_task-perception_run-01_bold_preproc.nii.gz') # A template nifti file

# Load Data
print('Loading Data...')
# correlations = get_test_correlations(matlab_file, test_image_ids, train_image_ids, images_npz, roi='ROI_VC')
# print(correlations)
# exit()
x_train, x_test, y_train, y_test, xyz = load_data(matlab_file, test_image_ids, train_image_ids, images_npz, roi='ROI_VC')
NUM_VOXELS = y_train.shape[1]
print(NUM_VOXELS)
# y_test = y_test * 0.5

# with open('models/' + model_name + '/train-history.json', 'r') as file:
#     train_hist = json.load(file)

# Load model
# print('Loading Model...')
#model = EncoderMach7(num_voxels=NUM_VOXELS, routing='dynamic', caps_act='squash')
# model = CapsEncoder(num_voxels=NUM_VOXELS, num_output_capsules=10)
# model = EncoderMach3(NUM_VOXELS)
# print(model.class_name)
# model.load_weights('../trained_models/' + model_name + '/model_weights').expect_partial()
# model.load_weights('../trained_models/' + model_name + '/best_ckpt').expect_partial()

# Calculating model predictions
# print('Getting Model Predictions...')
y_pred_gaziv = np.loadtxt(os.path.join(base_dir, 'trained_models/beliy/gaziv_y_pred.csv'), delimiter=',')
# y_pred = model.predict(x_test)
# np.savetxt(os.path.join(base_dir, 'trained_models', model_name, 'y_pred.csv'), y_pred, delimiter=',')
y_pred = np.loadtxt(os.path.join(base_dir, 'trained_models', model_name, 'y_pred.csv'), delimiter=',')
# y_pred = np.broadcast_to(np.mean(y_train, axis=0), shape=y_test.shape)
# y_pred = np.add(y_pred, np.random.exponential(0.04, y_pred.shape))

print('Finishing Up...')
correlation = []
g_correlation = []
voxel_corr = []
vc_p = []

for i in range(NUM_VOXELS):
    voxel_corr.append(stats.pearsonr(y_pred[:, i], y_test[:, i])[0])
    vc_p.append(stats.pearsonr(y_pred[:, i], y_test[:, i])[1])
for i in range(50):
    correlation.append(stats.pearsonr(y_test[i], y_pred[i])[0]) # returns correlation coefficient r and two tailed p-value
    #g_correlation.append(stats.pearsonr(y_test[i], y_pred_gaziv[i])[0])

# print(tf.shape(correlation))
# print(['{0:0.3f}'.format(i) for i in correlation])
voxel_corr = np.array(voxel_corr)
vc_p = np.array(vc_p)
print('Mean Sample Correlation: ', np.mean(correlation))

print('Mean Voxel Correlation: ', np.mean(voxel_corr))
print(np.mean(np.square(y_pred - y_test)))
cos = cosine_similarity(y_test, y_pred)
print(np.mean(cos))

stds = np.std(y_pred, axis=0)
stdtest = np.std(y_test, axis=0)
# print('std: ', np.mean(stds))

vc_sorted = np.sort(voxel_corr)
print(np.mean(vc_sorted[-100:]))
# print(vc_sorted)

# stds_norm = stds/stdtest
# vp = np.where(voxel_corr > 0)
# vn = np.where(voxel_corr < 0)

# vc_std = voxel_corr * stds
# vc_std_norm = voxel_corr * stds_norm
# print(np.min(vc_std))

# vt = np.where(vc_std > abs(np.min(vc_std)))
# vb = np.where(vc_std < abs(np.min(vc_std)))
# print(len(vt[0]))
# print(np.mean(vc_std[vt]))
# print(np.mean(vc_std_norm[vt]))
# print(np.mean(voxel_corr[vt]))
# print(np.mean(stds[vt]))
# print(len(np.where(vc_p < 0.001)[0]))
# plt.scatter(stds[vt], voxel_corr[vt], s=0.5)
# plt.scatter(stds[vb], voxel_corr[vb], s=0.5)
# plt.ylim(-0.6, 1)
# plt.xlim(0, 0.6)
# plt.show()

import matplotlib
matplotlib.rcParams.update({'font.size': 16})
#plt.figure(figsize=[6, 6])
# b_vc = np.loadtxt(os.path.join(base_dir, 'trained_models/beliy_vox_corr.csv'), delimiter=',')
# plt.scatter(b_vc, voxel_corr, s=0.5)
# lims = [-0.5, 1]
# plt.plot(lims, lims, '--', color='red', alpha=0.5, zorder=0)
# plt.axhline(0, color='black', alpha=0.5, linewidth=1)
# plt.axvline(0, color='black', alpha=0.5, linewidth=1)
# plt.xlabel('Beliy 2019')
# plt.ylabel('Capsule Encoder')
# plt.tight_layout()
# plt.show()

#Critical Values for 50 samples
# p=0.001 @ r=0.4514
# p=0.0001 r=0.5223

# for 20 samples
# p=0.001 @ r=0.6788


# b_vc = np.loadtxt(os.path.join(base_dir, 'trained_models/beliy/beliy_vox_corr.csv'), delimiter=',')
# diff = b_vc - voxel_corr
# avg = (b_vc + voxel_corr)/2
# # print(np.max(diff))
# idx = np.where(np.maximum(b_vc, voxel_corr) > 0.4514)
# l = (diff[idx] > 0).sum()/len(diff[idx])
# r = (diff[idx] < 0).sum()/len(diff[idx])
# # print(l, r)
# fig = plt.figure(figsize=(6, 6))
# gs = fig.add_gridspec(2, 2, width_ratios=(7, 1), height_ratios=(1.5, 7), hspace=0.1)
# # Create the Axes.
# ax = fig.add_subplot(gs[1, 0])
# ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
# ax_c = fig.add_axes([.78,.13,.04,.58])
# ax_histx.hist(diff[idx], bins=np.linspace(-1, 0, num=21), edgecolor='black', linewidth=0.5, color='tab:blue')
# ax_histx.hist(diff[idx], bins=np.linspace(0, 1, num=21), edgecolor='black', linewidth=0.5, color='royalblue')
# ax_histx.set_xlim(-1, 1)
# ax_histx.tick_params(axis='x', which='both',
#                 bottom=False, labelbottom=False) 
# ax_histx.set_yscale('log')
# ax_histx.set_ylabel('Voxel Count')
# ax_histx.text(-0.75, 8, '0.0', color='tab:blue')
# ax_histx.text(0.8, 15, '1.0', color='royalblue')

# hex = ax.hexbin(b_vc - voxel_corr, (b_vc + voxel_corr)/2, cmap='Blues', gridsize=30, bins='log')
# plt.colorbar(hex, cax=ax_c)
# ax.plot([-1.08, 0], [-0.09, 0.45], '--', color='red')
# ax.plot([0, 1.08], [0.45, -0.09], '--', color='red')
# ax.axhline(y=0, color='black', alpha=0.3, linewidth=0.7)
# ax.axvline(0, color='black', linestyle='-', alpha=0.3, linewidth=0.7)
# ax.set_xlim(-1, 1)
# ax.set_xlabel('Capsule <-> Beliy 2019 \n(p2 - p1)')
# ax.set_ylabel('Average Voxelwise Correlation \n(p1 + p2)/2')

# plt.savefig(os.path.join('figures', 'st_plot_kami.png'), bbox_inches='tight', facecolor='white')
# plt.show()

# plt.plot(mb)
# plt.plot(mc)
# plt.show()

# plt.hexbin(b_vc - voxel_corr, (b_vc + voxel_corr)/2, cmap='Blues', gridsize=30, bins='log')
# plt.axvline(0, color='black', linestyle='-', alpha=0.3, linewidth=0.7)
# plt.plot([-0.9, 0], [0, 0.45], '--', color='red')
# plt.plot([0, 0.9], [0.45, 0], '--', color='red')
# plt.xlim(-1, 1)
# plt.colorbar()
# plt.show()


# y_pred = y_pred.astype(np.float32)
# y_test = y_test.astype(np.float32)
# def combined_loss(y_true, y_pred):
#     return mean_squared_error(y_true, y_pred) +  0.1*cosine_similarity(y_true, y_pred)

# loss = combined_loss(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# cos = cosine_similarity(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)
# print('loss:', np.mean(loss))
# print('mse:', np.mean(mse))
# print('cos:', np.mean(cos))
# print('mae:', np.mean(mae))
# print(correlation[4])

# plt.figure(figsize=(15, 5))
# plt.subplot(1, 3, 1)
# plt.hexbin(stds, voxel_corr, cmap='Blues', gridsize=50, bins='log', extent=(0, 0.6, -0.6, 1))
# plt.title('std')
# plt.subplot(1, 3, 2)
# plt.hexbin(stds/stdtest, voxel_corr, cmap='Blues', gridsize=50, bins='log')
# plt.title('normalized')
# plt.subplot(1, 3, 3)
# plt.hexbin(stdtest, voxel_corr, cmap='Blues', gridsize=50, bins='log')
# plt.title('ground truth')
# plt.xlim(0, 0.6)
# plt.ylim(-0.6, 1)
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.7)
# plt.hexbin(stds, voxel_corr, cmap='Blues', gridsize=50, bins='log', vmax='50', extent=(0, 0.6, -0.6, 1))
# plt.ylim(-0.6, 1)
# plt.xlim(0, 0.6)
# plt.xlabel('Voxel Response Sensitivity')
# plt.ylabel('Voxelwise Correlation')
# plt.colorbar()
# # plt.savefig(os.path.join('models', model_name, 'corr_std.png'), bbox_inches='tight', facecolor='white')
# # plt.savefig(os.path.join('figures', 'mean_corr_std_imagenet.png'), bbox_inches='tight', facecolor='white')
# plt.tight_layout()
# plt.show()

# vc_std = voxel_corr * stds
# print(np.min(vc_std))
# print(np.sqrt(abs(np.min(vc_std))))

# vt1 = np.where(vc_std > abs(np.min(vc_std)))[0]
# np.savetxt(os.path.join('trained_encoders', model_name, 'idx_above_thresh.csv'), vt, delimiter=',')
# vt_beliy = np.loadtxt(os.path.join(base_dir, 'trained_models/beliy/idx_above_thresh.csv'), delimiter=',')
# vt = np.union1d(vt1, vt_beliy).astype(int)
# vb = np.where(vc_std < abs(np.min(vc_std)))[0]
# print(len(vt1))
# print(len(vt_beliy))
# print(len(vt))
# plt.scatter(stds[vt], voxel_corr[vt], s=0.5)
# plt.scatter(stds[vb], voxel_corr[vb], s=0.5)
# plt.ylim(-1, 1)
# plt.xlim(0, 1)
# plt.show()

# rho, phi = cart2pol(1 - stds[vt1], 1 - voxel_corr[vt1])
# phi = phi*(180/np.pi) - 45
# rho = (np.sqrt(2) - rho)/np.sqrt(2)
# print('rho:', np.mean(rho))
# print('phi:', np.mean(phi))
# print('%:', len(vt1)*100/NUM_VOXELS)
# print('vc_t:', np.mean(voxel_corr[vt1]))
# print('std_t:', np.mean(stds[vt1]))
# print('rwc: ', np.mean(np.sign(vc_std[vt1])*np.sqrt(abs(vc_std[vt1]))))

# rho, phi = cart2pol(1 - stds[vt], 1 - voxel_corr[vt])
# phi = phi*(180/np.pi) - 45
# rho = (np.sqrt(2) - rho)/np.sqrt(2)
# print('rho:', np.mean(rho))
# print('phi:', np.mean(phi))
# print('vc_t:', np.mean(voxel_corr[vt]))
# print('std_t:', np.mean(stds[vt]))
# print('rwc: ', np.mean(np.sign(vc_std[vt])*np.sqrt(abs(vc_std[vt]))))

# voxel_corr = np.array(voxel_corr)
# nifti = export_brain_image(stds, template=template, xyz=xyz)
# if not os.path.exists('nifti/' + model_name + '/'):
#     os.makedirs('nifti/' + model_name + '/')
# nibabel.save(nifti, 'nifti/' + model_name + '/sub03_voxel_std.nii.gz')
# nibabel.save(nifti, 'nifti/sub03_train_mean.nii.gz')
# nibabel.save(nifti, 'nifti/' + 'sub03_s4.nii.gz')

handler = kamitani_data_handler(matlab_file=matlab_file, test_img_csv=test_image_ids, train_img_csv=train_image_ids)
print(handler.get_meta_keys())

roi_names = ['ROI_V1', 'ROI_V2', 'ROI_V3', 'ROI_V4', 'ROI_LOC', 'ROI_FFA', 'ROI_PPA', 'ROI_LVC', 'ROI_HVC']
vc_rs = voxel_corr * stds
rwc = np.sign(vc_rs)*np.sqrt(abs(vc_rs))

vc_beliy = np.loadtxt(os.path.join(base_dir, 'trained_models/beliy/beliy_vox_corr.csv'), delimiter=',')
vc_rs_beliy = vc_beliy * np.std(y_pred_gaziv, axis=0)
rwc_beliy = np.sign(vc_rs_beliy)*np.sqrt(abs(vc_rs_beliy))

roi_metrics = {'rwc': [], 'vc': [], 'rs': []}
roi_metrics_beliy = {'rwc': [], 'vc': [], 'rs': []}
for roi in roi_names:
    select = handler.get_meta_field(roi).astype(bool)
    print(roi)
    print(np.mean(rwc[select]), np.mean(voxel_corr[select]), np.mean(stds[select]))
    roi_metrics['rwc'].append(rwc[select])
    roi_metrics_beliy['rwc'].append(rwc_beliy[select])
    roi_metrics['vc'].append(voxel_corr[select])
    roi_metrics_beliy['vc'].append(vc_beliy[select])
    roi_metrics['rs'].append(stds[select])
    roi_metrics_beliy['rs'].append(np.std(y_pred_gaziv, axis=0)[select])

roi_names = ['V1', 'V2', 'V3', 'V4', 'LOC', 'FFA', 'PPA', 'LVC', 'HVC']
# plt.bar(roi_names, np.mean(roi_metrics_beliy['vc']))
# plt.bar(roi_names, np.mean(roi_metrics['vc']))
# # plt.ylabel('Response Weighted Correlation')
# plt.ylabel('Voxelwise Correlation')
# plt.legend(['Beliy 2019', 'Capsule'])
# plt.tight_layout()
# plt.show()

plt.violinplot(roi_metrics_beliy['vc'])
plt.violinplot(roi_metrics['vc'])
ax = plt.gca()
ax.set_xticks(np.arange(1, len(roi_names) + 1), labels=roi_names)
# plt.ylabel('Response Weighted Correlation')
plt.ylabel('Voxelwise Correlation')
plt.tight_layout()
plt.show()
