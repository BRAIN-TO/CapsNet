# Public Imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
# Custom Imports
import layers as caps_layers


'''Models

Contains various model archetectures using capsule layers

In this file:
-CapsNet

TO-DO:
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
        self.primary = caps_layers.PrimaryCaps2D(num_channels=32, kernel_size=9, capsule_dim=8, strides=2, padding='valid', activation='squash')
        self.dense = caps_layers.DenseCaps(num_capsules=10, capsule_dim=16, routing='dynamic', activation='norm')

        self.decoder = keras.Sequential(
            [
                layers.Dense(512, activation='relu'),
                layers.Dense(1024, activation='relu'),
                layers.Dense(784, activation='sigmoid')
            ],
            name='decoder'
        )

    def call(self, inputs): # Define this method solely so that we can build model and print model summary
        # Propagate through encoder
        conv_out = self.conv(inputs)
        pose1, a1 = self.primary(conv_out)
        pose2, a2 = self.dense([pose1, a1])

        pose_masked = tools.mask_output_capsules(a2, pose2, weighted=False)
        
        # Reconstruct image
        decoder_input = tf.reshape(pose_masked, [-1, pose_shape[1] * pose_shape[2] * pose_shape[3]])
        recon_images = self.decoder(decoder_input)

        return  a2, pose2, recon_images
    
    @tf.function
    def train_step(self, data):
        x, y = data # x are input images, y are ohe labels

        # Forward propagation
        ##############################
        with tf.GradientTape() as tape:
            # Propagate through encoder
            conv_out = self.conv(x)
            pose1, a1 = self.primary(conv_out)
            pose2, a2 = self.dense([pose1, a1])

            pose_masked = tools.mask_output_capsules(y, pose2, weighted=False)

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

        # Update metrics
        self.compiled_metrics.update_state(y, a2)

        # Return loss and other metrics
        output_dict = {m.name : m.result() for m in self.metrics}
        output_dict['loss'] = loss
        return output_dict

    @tf.function
    def test_step(self, data):
        x, y = data # x are input images, y are ohe labels

        # Propagate through encoder
        conv_out = self.conv(x)
        pose1, a1 = self.primary(conv_out)
        pose2, a2 = self.dense([pose1, a1])

        pose_masked = tools.mask_output_capsules(a2, pose2, weighted=False)

        # Update metrics
        self.compiled_metrics.update_state(y, a2)

        # Reconstruct image
        decoder_input = tf.reshape(pose_masked, [-1, pose_shape[1] * pose_shape[2] * pose_shape[3]])
        recon_images = self.decoder(decoder_input)

        # Calculate loss
        x_flat = tf.reshape(x, [-1, tf.math.reduce_prod(tf.shape(x)[1:])]) # flatten input images
        loss = self.loss(a2, recon_images, x_flat, y)

        # Return loss and other metrics
        output_dict = {m.name : m.result() for m in self.metrics}
        output_dict['loss'] = loss
        return output_dict

    @tf.function(
        input_signature=[tf.TensorSpec(shape=(None, 10), dtype=tf.float32), tf.TensorSpec(shape=(None, 10, 16, 1), dtype=tf.float32)]
    )
    def reconstruct_image(self, capsule_activations, capsule_poses):
        pose_masked = tools.mask_output_capsules(capsule_activations, capsule_poses, weighted=False)

        # Reconstruct image
        decoder_input = tf.reshape(pose_masked, [-1, pose_shape[1] * pose_shape[2] * pose_shape[3]])
        recon_images = self.decoder(decoder_input)

        return recon_images