# Public Imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
# Custom Imports
import layers as caps_layers
import tools


'''Models

Contains various model archetectures using capsule layers

In this file:
-CapsNet
-MatrixCapsNet
-HybridCapsNet (Not Functional)
-CapsRecons

TO-DO:
'''
        
class CapsNet(keras.Model):
    '''The original capsule network designed for MNIST

    The capsule network from the original paper published in 2017

    1.Sabour, S., Frosst, N. & Hinton, G. E. Dynamic Routing Between Capsules. 
    in Advances in Neural Information Processing Systems 30 (eds. Guyon, I. et al.) 
    3856–3866 (Curran Associates, Inc., 2017).
    '''
    def __init__(self,):
        super(CapsNet, self).__init__()

        # Define layers
        self.conv = layers.Conv2D(filters=256, kernel_size=9, strides=(1, 1), padding='valid', activation='relu')
        self.primary = caps_layers.PrimaryCaps2D(num_channels=32, kernel_size=9, capsule_dim=8, strides=2, padding='valid', activation='squash')
        self.dense = caps_layers.DenseCaps(num_capsules=10, capsule_dim=16, routing='dynamic', activation='norm', name='class_capsules')

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
        pose_shape = pose_masked.shape
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
            pose_shape = pose_masked.shape
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
        pose_shape = pose_masked.shape # shape [batch_size, num_caps, caps_dim[0], caps_dim[1]]
        decoder_input = tf.reshape(pose_masked, [-1, pose_shape[1] * pose_shape[2] * pose_shape[3]])
        recon_images = self.decoder(decoder_input)

        # Calculate loss
        x_flat = tf.reshape(x, [-1, tf.math.reduce_prod(tf.shape(x)[1:])]) # flatten input images
        loss = self.loss(a2, recon_images, x_flat, y)

        # Return loss and other metrics
        output_dict = {m.name : m.result() for m in self.metrics}
        output_dict['loss'] = loss
        return output_dict

    @tf.function( # Have to decorate function so that it is not lost when saving model using model.save
        input_signature=[tf.TensorSpec(shape=(None, 10), dtype=tf.float32), tf.TensorSpec(shape=(None, 10, 16, 1), dtype=tf.float32)]
    )
    def reconstruct_image(self, capsule_activations, capsule_poses):
        pose_masked = tools.mask_output_capsules(capsule_activations, capsule_poses, weighted=False)

        # Reconstruct image
        pose_shape = pose_masked.shape
        decoder_input = tf.reshape(pose_masked, [-1, pose_shape[1] * pose_shape[2] * pose_shape[3]])
        recon_images = self.decoder(decoder_input)

        return recon_images

class MatrixCapsNet(keras.Model):
    '''The convolutional capsule network from the 2018 paper

    1.Hinton, G. E., Sabour, S. & Frosst, N. Matrix capsules with EM routing. in (2018).

    For now designed for the MNIST dataset, will be generalized soon
    '''
    def __init__(self,):
        super(MatrixCapsNet, self).__init__()

        # Warning: testing out conv caps with dynamic routing for now.

        # Create network layers
        self.conv = layers.Conv2D(32, kernel_size=5, strides=(2, 2), padding='same', activation='relu')
        self.primary = caps_layers.PrimaryCaps2D(32, kernel_size=1, capsule_dim=[4, 4], strides=1, padding='valid', activation='sigmoid')
        self.convcaps1 = caps_layers.ConvCaps2D(32, kernel_size=3, strides=2, capsule_dim=[4,4], routing='EM')
        self.convcaps2 = caps_layers.ConvCaps2D(32, kernel_size=3, strides=1, capsule_dim=[4,4], routing='EM')
        self.classcaps = caps_layers.DenseCaps(10, capsule_dim=[4, 4], routing='EM', name='class_capsules', add_coordinates=True, pose_coords=[[0, 3], [1, 3]])

    def call(self, inputs):
        filters = self.conv(inputs)
        pose1, a1 = self.primary(filters)
        pose2, a2 = self.convcaps1([pose1, a1])
        pose3, a3 = self.convcaps2([pose2, a2])
        pose_out, a_out = self.classcaps([pose3, a3])

        return pose_out, a_out

    @tf.function
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            # Forward propagation
            filters = self.conv(x)
            pose1, a1 = self.primary(filters)
            pose2, a2 = self.convcaps1([pose1, a1])
            pose3, a3 = self.convcaps2([pose2, a2])
            pose_out, a_out = self.classcaps([pose3, a3])

            # Calculate losss
            loss = self.loss(y, a_out)

        # Calculate gradients
        training_vars = self.trainable_variables
        gradients = tape.gradient(loss, training_vars)

        # Optimize weights
        self.optimizer.apply_gradients(zip(gradients, training_vars))

        # Update metrics
        self.compiled_metrics.update_state(y, a_out)

        # Return loss and other metrics
        output_dict = {m.name : m.result() for m in self.metrics}
        output_dict['loss'] = loss
        return output_dict


    @tf.function
    def test_step(self, data):
        x, y = data

        # Forward propagation
        filters = self.conv(x)
        pose1, a1 = self.primary(filters)
        pose2, a2 = self.convcaps1([pose1, a1])
        pose3, a3 = self.convcaps2([pose2, a2])
        pose_out, a_out = self.classcaps([pose3, a3])

        # Calculate losss
        loss = self.loss(y, a_out)

        # Update metrics
        self.compiled_metrics.update_state(y, a_out)

        # Return loss and other metrics
        output_dict = {m.name : m.result() for m in self.metrics}
        output_dict['loss'] = loss
        return output_dict

class HybridCapsNet(MatrixCapsNet):
    '''
    A Matrix CapsNet that uses Dynamic Routing instead of EM Routing
    '''
    def __init__(self,):
        super(HybridCapsNet, self).__init__()

        # Loss at each step does not change
            # Probably due to diff between a_t and a_i being near zero
        # y_pred values all 0.9999+ (activation for each capsule near 1)
        # accuracy does vary
        # input_poses to convcaps potentially approaching zero, but otherwise nothing weird
        # nothing noticeably weird with kernel weights of convcaps
        # capsule_pose output after routing definitely approaches zero


        # Create network layers
        self.conv = layers.Conv2D(32, kernel_size=5, strides=(2, 2), padding='same', activation='relu')
        self.primary = caps_layers.PrimaryCaps2D(32, kernel_size=1, capsule_dim=[4, 4], strides=1, padding='valid', activation='squash')
        self.convcaps1 = caps_layers.ConvCaps2D(32, kernel_size=3, strides=2, capsule_dim=[4,4], routing='dynamic', activation='squash')
        self.convcaps2 = caps_layers.ConvCaps2D(32, kernel_size=3, strides=1, capsule_dim=[4,4], routing='dynamic', activation='squash')
        self.classcaps = caps_layers.DenseCaps(10, capsule_dim=[4, 4], routing='dynamic', name='class_capsules', activation='norm', add_coordinates=True, pose_coords=[[0, 3], [1, 3]])

class CapsRecon(CapsNet):
    '''Same model as CapsNet except only using reconstruction loss
    '''

    # Got similar loss on test set after 2 epochs as CapsNet did after 
    # 50 epochs (0.0219 vs 0.0191)
    # Compared reconstructed images to og images and they looked good in general
    # Adjusted dimensions and features appear to have been learned
    def __init__(self):
        super(CapsRecon, self).__init__()

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
            pose_shape = pose_masked.shape
            decoder_input = tf.reshape(pose_masked, [-1, pose_shape[1] * pose_shape[2] * pose_shape[3]])
            recon_images = self.decoder(decoder_input)

            # Calculate loss
            x_flat = tf.reshape(x, [-1, tf.math.reduce_prod(tf.shape(x)[1:])]) # flatten input images
            loss = self.loss(x_flat, recon_images)
        
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
        pose_shape = pose_masked.shape # shape [batch_size, num_caps, caps_dim[0], caps_dim[1]]
        decoder_input = tf.reshape(pose_masked, [-1, pose_shape[1] * pose_shape[2] * pose_shape[3]])
        recon_images = self.decoder(decoder_input)

        # Calculate loss
        x_flat = tf.reshape(x, [-1, tf.math.reduce_prod(tf.shape(x)[1:])]) # flatten input images
        loss = self.loss(x_flat, recon_images)

        # Return loss and other metrics
        output_dict = {m.name : m.result() for m in self.metrics}
        output_dict['loss'] = loss
        return output_dict