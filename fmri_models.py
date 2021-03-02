# Public Imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
# Custom Imports
import layers as caps_layers
import tools

'''FMRI Models

Contains models for fmri data
'''

class CapsEncoder(keras.Model):
    '''The original capsule network modified for the generic object decoding dataset
    '''
    def __init__(self, num_voxels):
        super(CapsEncoder, self).__init__()

        # Define layers
        self.conv = layers.Conv2D(filters=256, kernel_size=9, strides=(1, 1), padding='valid', activation='relu')
        self.primary = caps_layers.PrimaryCaps2D(num_channels=32, kernel_size=9, capsule_dim=8, strides=2, padding='valid', activation='squash')
        self.dense = caps_layers.DenseCaps(num_capsules=10, capsule_dim=16, routing='dynamic', activation='norm', name='class_capsules')

        self.decoder = keras.Sequential(
            [
                layers.Dense(512, activation='relu'),
                layers.Dense(1024, activation='relu'),
                layers.Dense(num_voxels, activation='sigmoid')
            ],
            name='decoder'
        )
    
    def call(self, inputs): # Define this method solely so that we can build model and print model summary
        # Propagate through encoder
        conv_out = self.conv(inputs)
        pose1, a1 = self.primary(conv_out)
        pose2, a2 = self.dense([pose1, a1])
        
        # Reconstruct image
        pose_shape = pose2.shape # [batch_size, num_caps, caps_dim[0], caps_dim[1]]
        decoder_input = tf.reshape(pose2, [-1, pose_shape[1] * pose_shape[2] * pose_shape[3]])
        recon_images = self.decoder(decoder_input)

        return recon_images

    def train_step(self, data):
        x, y = data # x are input images, y is fmri voxels

        # Forward propagation
        ##############################
        with tf.GradientTape() as tape:
            # Propagate through encoder
            conv_out = self.conv(x)
            pose1, a1 = self.primary(conv_out)
            pose2, a2 = self.dense([pose1, a1])

            # Reconstruct image
            pose_shape = pose2.shape
            decoder_input = tf.reshape(pose2, [-1, pose_shape[1] * pose_shape[2] * pose_shape[3]])
            recon_images = self.decoder(decoder_input)

            # Calculate loss
            loss = self.loss(y, recon_images)
        
        # Calculate gradients
        training_vars = self.trainable_variables
        gradients = tape.gradient(loss, training_vars)

        # Optimize weights
        self.optimizer.apply_gradients(zip(gradients, training_vars))

        # Return loss and other metrics
        output_dict = {m.name : m.result() for m in self.metrics}
        output_dict['loss'] = loss
        return output_dict

    def test_step(self, data):
        x, y = data # x are input images, y are ohe labels

        # Propagate through encoder
        conv_out = self.conv(x)
        pose1, a1 = self.primary(conv_out)
        pose2, a2 = self.dense([pose1, a1])

        # Reconstruct image
        pose_shape = pose2.shape
        decoder_input = tf.reshape(pose2, [-1, pose_shape[1] * pose_shape[2] * pose_shape[3]])
        recon_images = self.decoder(decoder_input)

        # Calculate loss
        loss = self.loss(y, recon_images)

        # Return loss and other metrics
        output_dict = {m.name : m.result() for m in self.metrics}
        output_dict['loss'] = loss
        return output_dict

class MatrixCapsEncoder(keras.Model):
    '''The Matrix Capsule Network from the original paper on EM routing
    modified for fmri data
    '''

    def __init__(self, num_voxels):
        super(MatrixCapsEncoder, self).__init__()

        # Create network layers
        self.conv = layers.Conv2D(32, kernel_size=5, strides=(2, 2), padding='same', activation='relu')
        self.primary = caps_layers.PrimaryCaps2D(32, kernel_size=1, capsule_dim=[4, 4], strides=1, padding='valid', activation='sigmoid')
        self.convcaps1 = caps_layers.ConvCaps2D(32, kernel_size=3, strides=2, capsule_dim=[4,4], routing='EM')
        self.convcaps2 = caps_layers.ConvCaps2D(32, kernel_size=3, strides=1, capsule_dim=[4,4], routing='EM')
        self.classcaps = caps_layers.DenseCaps(10, capsule_dim=[4, 4], routing='EM', name='class_capsules', add_coordinates=True, pose_coords=[[0, 3], [1, 3]])

        self.decoder = keras.Sequential(
            [
                layers.Dense(512, activation='relu'),
                layers.Dense(1024, activation='relu'),
                layers.Dense(num_voxels, activation='sigmoid')
            ],
            name='decoder'
        )

    def call(self, inputs):
        # Propagate through capslayers
        filters = self.conv(inputs)
        pose1, a1 = self.primary(filters)
        pose2, a2 = self.convcaps1([pose1, a1])
        pose3, a3 = self.convcaps2([pose2, a2])
        pose4, a4 = self.classcaps([pose3, a3])

        # Reconstruct image
        pose_shape = pose4.shape # [batch_size, num_caps, caps_dim[0], caps_dim[1]]
        decoder_input = tf.reshape(pose4, [-1, pose_shape[1] * pose_shape[2] * pose_shape[3]])
        fmri_pred = self.decoder(decoder_input)

        return fmri_pred

    def train_step(self, data):
        print('train_step started')
        x, y = data

        with tf.GradientTape() as tape:
            # Propagate through capslayers
            filters = self.conv(x)
            pose1, a1 = self.primary(filters)
            pose2, a2 = self.convcaps1([pose1, a1])
            pose3, a3 = self.convcaps2([pose2, a2])
            pose4, a4 = self.classcaps([pose3, a3])
            # Reconstruct image
            pose_shape = pose4.shape # [batch_size, num_caps, caps_dim[0], caps_dim[1]]
            decoder_input = tf.reshape(pose4, [-1, pose_shape[1] * pose_shape[2] * pose_shape[3]])
            fmri_pred = self.decoder(decoder_input)

            loss = self.loss(y, fmri_pred)

        # Calculate gradients
        training_vars = self.trainable_variables
        gradients = tape.gradient(loss, training_vars)

        # Optimize weights
        self.optimizer.apply_gradients(zip(gradients, training_vars))
        
        # Return loss and other metrics
        output_dict = {m.name : m.result() for m in self.metrics}
        output_dict['loss'] = loss
        return output_dict

    def test_step(self, data):
        x, y = data

        filters = self.conv(x)
        pose1, a1 = self.primary(filters)
        pose2, a2 = self.convcaps1([pose1, a1])
        pose3, a3 = self.convcaps2([pose2, a2])
        pose4, a4 = self.classcaps([pose3, a3])

        # Reconstruct image
        pose_shape = pose4.shape # [batch_size, num_caps, caps_dim[0], caps_dim[1]]
        decoder_input = tf.reshape(pose4, [-1, pose_shape[1] * pose_shape[2] * pose_shape[3]])
        fmri_pred = self.decoder(decoder_input)

        loss = self.loss(y, fmri_pred)
        
        # Return loss and other metrics
        output_dict = {m.name : m.result() for m in self.metrics}
        output_dict['loss'] = loss
        return output_dict