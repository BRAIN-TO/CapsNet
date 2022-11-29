from misc.kamitani_data_handler import kamitani_data_handler
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import itertools
from scipy import stats
import math


def load_data(matlab_file, test_img_ids, train_img_ids, images_npz, roi='ROI_VC'):
    '''
    Loads data for model

    Args:
        matlab_file: A matlab file containing the fmri data for a given subject
        test_img_ids: A csv file containing the numerical identifiers for each test image
        train_img_ids: A csv file containing the numerical identifiers for each train image
        images_npz: An npz file that contains the actual images
        roi: The roi to select

    Returns
        x_train: training images
        x_test: testing images
        y_train: fmri activations for training images
        y_test: average fmri activations for testing images
        xyz: voxel coordinates for fmri datas
    '''
    
    data_handler = kamitani_data_handler(matlab_file=matlab_file, test_img_csv=test_img_ids, train_img_csv=train_img_ids)
    train_fmri, test_fmri, test_fmri_avg = data_handler.get_data(roi=roi)
    NUM_VOXELS = train_fmri.shape[1]
    y_train = train_fmri
    y_test = test_fmri_avg

    npz_file = np.load(images_npz)
    test_images = npz_file['test_images']
    train_images = npz_file['train_images']

    train_labels, test_labels = data_handler.get_labels() # Returns the labels for each image

    x_test = test_images # test_fmri_avg is already in order, hence do not need to reorder x_test
    y_test = test_fmri_avg # For the test set, we only want one sample per test image, hence we use fmri averages across runs

    x_train = train_images[train_labels]

    '''
    Note that each image is presented multiple times. Therefore there are more 
    samples than there are images for both the training and test image sets

    train_fmri: The preprocessed fmri data for the training images. 
    test_fmri: The preprocessing fmri data for testing images
    test_fmri_avg: The preprocessed fmri data of the test set averaged across
        all trials for each test image. (Ie the average activations for an image
        after multiple trials)

        shape: [samples, voxels]

    note that each voxel is scaled based on the mean and std of the entire dataset
    '''

    # Convert images back to 255 scale and subtract mean
    MEAN_PIXELS = [123.68, 116.779, 103.939]
    means = tf.constant(MEAN_PIXELS, shape=[1, 1, 1, 3], dtype=tf.float32)
    x_test = tf.subtract(x_test*255.0, means)
    x_train = tf.subtract(x_train*255.0, means)

    # renormalize data
    # scaler = MinMaxScaler()
    # scaler.fit(x_train)
    # x_train = scaler.transform(x_train)
    # x_test = scaler.transform(x_test)

    xyz, _ = data_handler.get_voxel_loc(roi=roi)


    return [x_train, x_test, y_train, y_test, xyz]

def get_test_correlations(matlab_file, test_img_ids, train_img_ids, images_npz, roi='ROI_VC'):
    '''
    Calculates average correlation between runs for each test stimuli presentation

    Args:
        matlab_file: A matlab file containing the fmri data for a given subject
        test_img_ids: A csv file containing the numerical identifiers for each test image
        train_img_ids: A csv file containing the numerical identifiers for each train image
        images_npz: An npz file that contains the actual images
        roi: The roi to select

    Returns:
        correlations: The average correlation between each stimuli presentation
            for each test stimuli
    '''

    data_handler = kamitani_data_handler(matlab_file=matlab_file, test_img_csv=test_img_ids, train_img_csv=train_img_ids)
    train_fmri, test_fmri, test_fmri_avg = data_handler.get_data(roi='ROI_VC')
    NUM_VOXELS = train_fmri.shape[1]

    npz_file = np.load(images_npz)
    test_images = npz_file['test_images']
    train_images = npz_file['train_images']

    train_labels, test_labels = data_handler.get_labels() # Returns the labels for each image in train_fmri and test_fmri respectively

    num_test_images = max(test_labels) + 1 # Test labels go from 0 to 49 (50 different test images)
    correlations = []
    for i in range(num_test_images):
        fmri_group = test_fmri[test_labels==i] # Retrieves all the fmri runs for a single test image
        group_correlations = []

        # Get correlation between all test runs for one image
        # for fmri_pair in itertools.combinations(fmri_group, r=2):
        #     correlation = stats.pearsonr(fmri_pair[0], fmri_pair[1])[0] # returns both correlation coef and p-value ignore p-value
        #     group_correlations.append(correlation)
        # avg = np.mean(group_correlations)
        

        # Get correlations for all test runs with avg
        # for i, fmri in enumerate(fmri_group):
        #     correlation = stats.pearsonr(fmri, test_fmri_avg[i])[0]
        #     group_correlations.append(correlation)
        # avg = np.mean(group_correlations)
        

        # get correlation between two sub averages
        avg1 = np.mean(fmri_group[:len(fmri_group)//2], axis=0)
        avg2 = np.mean(fmri_group[len(fmri_group)//2:], axis=0)
        avg = stats.pearsonr(avg1, avg2)[0]

        correlations.append(avg)
    return correlations

def remove_low_variance_voxels(fmri_set1, fmri_set2, xyz, threshold):
    '''Removes voxels with low variance from the data

    Calculates the variance for each voxel using the firsts set of data and uses
    a threshold to remove voxels based on their variance in both datasets

    Args:
        fmri_set1: fmri data from which variance will be calculated for threshold. [samples, voxels]
        fmri_set2: fmri data to remove corresponding voxels from. [samples, voxels]
        xyz: the coordinates for each voxel. Should had shape [num_dim, num_voxels]
        threshold (int): The threshold variance. Voxels with variance below
            this value will be removed

    returns:
        fmri_set1_new: fmri data with low variance voxels removed
        fmri_set2_new: fmri data with the corresponding voxels removed
        xyz_new: The coordinates of the high variance voxels. 
    '''

    var = np.var(fmri_set1, axis=0)
    high_var_bool = var > threshold # Get indices of where high variance voxels are
    fmri_set1_new = fmri_set1[:, high_var_bool]
    fmri_set2_new = fmri_set2[:, high_var_bool]
    xyz_new = xyz[:, high_var_bool]
    
    return fmri_set1_new, fmri_set2_new, xyz_new
    
def cart2pol(x, y):
    '''Converts a vector in cartesian coordinates to polar coordinates

    Args:
        x: x_component of vector
        y: y_component of vector

    Returns:
        vec: Tuple containing magnitude and angle (in radians)  of vector
            in that order
    '''
    rho = np.sqrt(np.square(x) + np.square(y))
    phi = np.arctan2(y, x)
    return (rho, phi)

def pol2cart(rho, phi):
    '''Converts a vector in polar coordinates to cartesian coordinate

    Args:
        rho: Magnitude of vector
        phi: direction of vector in radians

    Returns:
        vec: Tuple containing x and y component of vector in  that order
    '''
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)