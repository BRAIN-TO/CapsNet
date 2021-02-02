import tensorflow as tf
import tools

'''Losses

Contains the various loss functions used for capsule networks

In this file:
-
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

    a_t = tf.reshape(tf.boolean_mask(y_pred, mask_t), shape=[shape[0], 1]) # activation of correct class
    a_i = tf.reshape(tf.boolean_mask(y_pred, mask_i), shape=[shape[0], shape[1] - 1]) # Activations of incorrect classes

    loss = tf.reduce_sum(tf.square(tf.maximum(0.0, margin - (a_t - a_i))))

    return loss



    
