# PubliC API's
import tensorflow as tf
# Custom Imports
import pycaps.tools as tools
from tensorflow.keras.losses import cosine_similarity, mean_squared_error, Loss, Reduction

'''Losses

Contains the various loss functions used for capsule networks

In this file:
-margin_recon_loss
-spread_loss
-mse_cosine_loss
'''

def margin_recon_loss(capsule_activations, reconstructed_images, input_images, labels, m_plus=0.9, m_minus=0.1, lambda_val=0.5, alpha=0.0005):
    '''Uses margin loss and mse reconstruction loss

    The original loss function used in the 2017 paper Dynamic Routing
    between capsules. Meant for a categorical classifier capsule network
    where the output capsules represent classes

    Args:
        capule_activations (tensor): A tensor with the activations of the
            final layer of class capsules [batch_size, num_capsules]
        reconstructed_images (tensor): The reconstructed images. One for 
            each input image.
        input_images (tensor): The ground truth images used as inputs
        labels (tensor): The class labels for the images. Must correspond
            to class capsule activations (should be one hot encoded)
        m_plus (float): Value between zero and one, level of confidence 
            required for a completely correct prediction
        m_minus (float): Value between zero and one, lack of confidence
            required for a completely incorrect prediction
        lambda_val (float): Regularization factor for loss contributed by
            incorrect predictions
        alpha (float): Regularization factor for loss contributed by the
            reconstruction error
    '''
    assert len(reconstructed_images.shape) < 4, 'Reconstructed images' \
        'tensor should have 2 or 3 dimensions'
    assert len(input_images.shape) < 4, 'Input images' \
        'tensor should have 2 or 3 dimensions'


    preds = capsule_activations
    
    # Calculate margin loss
    left_margin = tf.square(tf.maximum(0.0, m_plus - preds))
    right_margin = tf.square(tf.maximum(0.0, preds - m_minus))

    margin_loss = tf.add(labels * left_margin, lambda_val * (1 - labels) * right_margin) # Add and weight both sides of loss equation
    margin_loss = tf.reduce_mean(tf.reduce_sum(margin_loss, axis=-1)) # Sum of loss for each digit, average loss for whole batch

    # Check if recon and input images have been flattened already
    if len(reconstructed_images.shape) == 3:
        shape = tf.shape(reconstructed_images)
        reconstructed_images = tf.reshape(reconstructed_images, [-1, shape[1]*shape[2]])

    if len(input_images.shape) == 3:
        shape = tf.shape(input_images)
        input_images = tf.reshape(input_images, [-1, shape[1]*shape[2]])

    # Get MSE recon loss
    recon_loss = tf.reduce_mean(tf.square(input_images - reconstructed_images)) # Reduce mean takes mean for pixels and batches

    # Get total loss
    loss = tf.add(margin_loss, alpha * recon_loss)
    
    # loss should be a single value
    return loss

def spread_loss(y, y_pred, margin=0.2):
    '''The spread loss function from the 2018 paper 'Matrix Capsules with EM Routing'

    Args:
        y (tensor): Ground truth categorical labels. One hot encoded
            shape: [batch_size, num_classes]
        y_pred (tensor): Predicted labels shape: [batch_size, num_classes]
    '''
    shape = y.shape
    mask_t = tf.equal(y, 1) # Locations of correct labels
    mask_i = tf.equal(y, 0) # Locations of incorrect labels

    # Seperate activations predicted class from other classes
    a_t = tf.reshape(tf.boolean_mask(y_pred, mask_t), shape=[-1, 1]) # activation of correct class
    a_i = tf.reshape(tf.boolean_mask(y_pred, mask_i), shape=[-1, shape[1] - 1]) # Activations of incorrect classes

    # Calculate spread loss
    loss = tf.reduce_sum(tf.square(tf.maximum(0.0, margin - (a_t - a_i))))

    return loss

def mse_cosine_loss(y, y_pred, mse_factor=1, cosine_factor=0.1):
    mse = mean_squared_error(y, y_pred)
    cos = cosine_similarity(y, y_pred)
    return mse_factor*mse + cosine_factor*cos

def pearson_r_approx(y_true, y_pred, axis=0):
    '''Calculates an approximation of the correlation between between model 
    predictions and ground truth using the cosine similarity function.

    Args:
        y_pred (Tensor): The model predictions. A tensor with shape 
            (batch_size, D) where D is the dimensionality of the output vector
        y_test (Tensor): The ground truth values. A tensor with shape 
            (batch_size, D) where D is the dimensionality of the output vector
        axis (int): Which axis reduce when computing the correlations. If 0, 
            computes correlations between vectors of length (batch_size). If 1, 
            computes correlations between vectors of length (D)

    Returns:
        corrs (Tensor): A tensor containing all the correlation coefficients for
            the batch. If axis is 0, corrs shape will be (D,), if axis is 1 corrs
            shape will be (batch_size,)
    '''
    x = y_true
    y = y_pred
    mx = tf.reduce_mean(x, axis=axis, keepdims=True)
    my = tf.reduce_mean(y, axis=axis, keepdims=True)
    xm, ym = x - mx, y - my
    t1_norm = tf.nn.l2_normalize(xm, axis = axis)
    t2_norm = tf.nn.l2_normalize(ym, axis = axis)
    cosine = -1*tf.losses.cosine_similarity(t1_norm, t2_norm, axis = axis)
    return cosine

def pearson_r(y_true, y_pred, axis=0):
    '''Calculates the exact correlation between between model predictions and 
    ground truth

    Args:
        y_pred (Tensor): The model predictions. A tensor with shape 
            (batch_size, D) where D is the dimensionality of the output vector
        y_test (Tensor): The ground truth values. A tensor with shape 
            (batch_size, D) where D is the dimensionality of the output vector
        axis (int): Which axis reduce when computing the correlations. If 0, 
            computes correlations between vectors of length (batch_size). If 1, 
            computes correlations between vectors of length (D)

    Returns:
        corrs (Tensor): A tensor containing all the correlation coefficients for
            the batch. If axis is 0, corrs shape will be (D,), if axis is 1 corrs
            shape will be (batch_size,)
    '''
    x = y_true
    y = y_pred
    mx = tf.reduce_mean(x, axis=axis, keepdims=True)
    my = tf.reduce_mean(y, axis=axis, keepdims=True)
    xm, ym = x - mx, y - my
    numerator = tf.reduce_sum(tf.multiply(xm, ym), axis=axis)
    denominator = tf.sqrt(tf.multiply(tf.reduce_sum(xm**2, axis=axis), tf.reduce_sum(ym**2, axis=axis)))
    return tf.divide(numerator, denominator)

class MSE_Vox_Corr(Loss):
    def __init__(self, mse_factor=0.5, vc_factor=0.5):
        '''Change reduction type to sum since using sum_over_batch will scale things
        incorrectly when using multiple replicas. This is because we don't have a single
        loss value for each input hence call does not return a tensor of shape (batch_size, d0, ..., dN)
        '''
        super(MSE_Vox_Corr, self).__init__(reduction=Reduction.SUM)
        self.mse_factor = mse_factor
        self.vc_factor = vc_factor

    def call(self, y_true, y_pred):
        mse = tf.reduce_mean(tf.math.square(y_pred - y_true), axis=None)
        vc = tf.reduce_mean(pearson_r(y_true, y_pred, axis=1))
        # we want to minimize mse and maximice vc
        return self.mse_factor*mse - self.vc_factor*vc

    def get_config(self):
        cfg = super().get_config()
        cfg['mse_factor'] = self.mse_factor
        cfg['vc_factor'] = self.vc_factor
        return cfg


# def margin_loss(y, y_pred, m_plus=0.9, m_minus=0.1, lambda_val=0.5):
#     preds = y_pred
    
#     # Calculate margin loss
#     left_margin = tf.square(tf.maximum(0.0, m_plus - preds))
#     right_margin = tf.square(tf.maximum(0.0, preds - m_minus))

#     margin_loss = tf.add(y * left_margin, lambda_val * (1 - y) * right_margin) # Add and weight both sides of loss equation
#     margin_loss = tf.reduce_mean(tf.reduce_sum(margin_loss, axis=-1)) # Sum of loss for each digit, average loss for whole batch
#     return margin_loss




