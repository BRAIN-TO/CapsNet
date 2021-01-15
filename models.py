import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import layers as caps_layers

'''Models

Contains various model archetectures using capsule layers

In this file:
-CapsNet
'''

class CapsNet(keras.Model):
    '''The original CapsNet model designed for MNIST

    The capsule network from the original paper published in 2017

    S. Sabour, N. Frosst, and G. E. Hinton, “Dynamic Routing Between Capsules,” 
    in Advances in Neural Information Processing Systems 30, Long Beach, 
    California, 2017, pp. 3856–3866, Accessed: Oct. 15, 2020.
    '''
    def __init__(self,):
        '''The original capsule network

        From the 2017 paper 'Dynamic Routing Between Capsules'

        No input arguments as this model has the exact same parameters
        as the model from the paper
        '''
        super(CapsNet, self).__init__()

        # Later check to see if these will work with keras.Sequential
        self.conv = layers.Conv2D(filters=256, kernel_size=9, strides=(1, 1), padding='valid', activation='relu')
        self.primary = caps_layers.PrimaryCaps2D(num_channels=32, kernel_size=9, capsule_dim=8, stride=2, padding='valid', activation='squash')
        self.dense = caps_layers.DenseCaps(num_capsules=10, capsule_dim=16, routing='dynamic', activation='squash')

        self.decoder = keras.Sequential(
            [
                layers.Dense(512, activation='relu'),
                layers.Dense(1024, activation='relu'),
                layers.Dense(784, activation='sigmoid')
            ],
            name='decoder'
        )
    
    def train_step(self, data):
        x, y = data # x are input images, y are ohe labels

        # Forward propagation
        ##############################
        with tf.GradientTape() as tape:
            # Propagate through encoder
            conv_out = self.conv(x)
            pose1, a1 = self.primary(conv_out)
            pose2, a2 = self.dense([pose1, a1])

            # Mask outputs
            pose_shape = pose2.shape # pose shape [batch_size, num_capsules] + caps_dim
            mask = tf.expand_dims(y, axis=-1) # mask shape [batch_size, num_classes=num_capsules, 1]
            pose_flat = tf.reshape(pose2, [-1, pose_shape[1], pose_shape[-2] * pose_shape[-1]]) # flatten pose matrices into vectors
            pose_masked = tf.multiply(pose_flat, mask) # shape [batch_size, num_capsules] + caps_dim

            # Reconstruct image
            decoder_input = tf.reshape(pose_masked, [-1, pose_shape[1] * pose_shape[2] * pose_shape[3]])
            recon_images = self.decoder(decoder_input)

            # Calculate loss
            x_flat = tf.reshape(x, [-1, tf.math.reduce_prod(tf.shape(x)[1:])]) # flatten input images
            loss = self.loss(a2, recon_images, x_flat, y)
        
        # Calculate gradients
        training_vars = self.trainable_variables
        gradients = tape.gradient(loss, training_vars)

        # Optimize weights
        self.optimizer.apply_gradients(zip(gradients, training_vars))

        # Update accuracy
        self.compiled_metrics.update_state(y, a2)

        return {m.name : m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data # x are input images, y are ohe labels

        # Propagate through encoder
        conv_out = self.conv(x)
        pose1, a1 = self.primary(conv_out)
        pose2, a2 = self.dense([pose1, a1])

        # Mask outputs with predicted classes 
        pose_shape = pose2.shape # pose shape [batch_size, num_capsules] + caps_dim
        mask = tf.expand_dims(y, axis=-1) # mask shape [batch_size, num_classes=num_capsules, 1]
        pose_flat = tf.reshape(pose2, [-1, pose_shape[1], pose_shape[-2]*pose_shape[-1]]) # flatten pose matrices into vectors
        pose_masked = tf.multiply(pose_flat, mask) # shape [batch_size, num_capsules] + caps_dim

        # Update accuracy
        self.compiled_metrics.update_state(y, a2)

        return {m.name : m.result() for m in self.metrics}


    def predict(self, data, return_poses=False, return_reconstructions=False):
        x, y = data # x are input images, y are ohe labels

        # Propagate through encoder
        conv_out = self.conv(x)
        pose1, a1 = self.primary(conv_out)
        pose2, a2 = self.dense([pose1, a1])

        # Mask outputs with predicted classes 
        pose_shape = pose2.shape # pose shape [batch_size, num_capsules] + caps_dim
        mask = tf.expand_dims(y, axis=-1) # mask shape [batch_size, num_classes=num_capsules, 1]
        pose_flat = tf.reshape(pose2, [-1, pose_shape[1], pose_shape[-2]*pose_shape[-1]]) # flatten pose matrices into vectors
        pose_masked = tf.multiply(pose_flat, mask) # shape [batch_size, num_capsules] + caps_dim

        if return_reconstructions == True:
            # Reconstruct image
            decoder_input = tf.reshape(pose_masked, [-1, pose_shape[1] * pose_shape[2] * pose_shape[3]])
            recon_images = self.decoder(decoder_input)
            
            if return_poses == True:
                return  a2, pose2, recon_images
            else:
                return a2, recon_images
        elif return_poses == True:
            return a2, pose2
        else:
            return a2