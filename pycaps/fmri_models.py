# Public Imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import time
import scipy
# Custom Imports
import pycaps.layers as caps_layers
import pycaps.tools as tools

'''FMRI Models

Contains models for fmri data
'''

class CapsEncoder(keras.Model):
    '''The original capsule network modified to be an encoder
    '''
    def __init__(self, num_voxels, output_activation='linear', num_output_capsules=10, routing='dynamic', caps_act='squash'):
        super(CapsEncoder, self).__init__()
        self.class_name = 'CapsEncoder'

        # Define layers
        self.conv = layers.Conv2D(filters=256, kernel_size=9, strides=(1, 1), padding='valid', activation='relu')
        self.primary = caps_layers.PrimaryCaps2D(num_channels=32, kernel_size=9, capsule_dim=8, strides=2, padding='valid', activation=caps_act)
        self.dense = caps_layers.DenseCaps(num_capsules=num_output_capsules, capsule_dim=16, routing=routing, activation='squash', name='class_capsules')

        self.decoder = keras.Sequential(
            [
                layers.Dense(512, activation='relu'),
                layers.Dense(1024, activation='relu'),
                layers.Dense(num_voxels, activation=output_activation)
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
        y_pred = self.decoder(decoder_input)

        return y_pred

    def train_step(self, data):
        x, y = data # x are input images, y is fmri voxels

        # Forward propagation
        ##############################
        with tf.GradientTape() as tape:
            y_pred = self.call(x)

            # Calculate loss
            loss = self.compiled_loss(y, y_pred)
        
        # Calculate gradients
        training_vars = self.trainable_variables
        gradients = tape.gradient(loss, training_vars)

        # Optimize weights
        self.optimizer.apply_gradients(zip(gradients, training_vars))

        # Return loss and other metrics
        self.compiled_metrics.update_state(y, y_pred)
        output_dict = {m.name : m.result() for m in self.metrics}
        return output_dict

    def test_step(self, data):
        x, y = data # x are input images, y are ohe labels

        y_pred = self.call(x)

        # Calculate loss
        loss = self.compiled_loss(y, y_pred)

        # Return loss and other metrics
        self.compiled_metrics.update_state(y, y_pred)
        output_dict = {m.name : m.result() for m in self.metrics}
        return output_dict

    def get_config(self):
        output_dict = {}
        layer_configs = []
        output_dict['decoder_config'] = self.decoder.get_config()
        layer_configs.append({'class_name': 'Conv2D', 'config': self.conv.get_config()})
        layer_configs.append({'class_name': 'PrimaryCaps2D', 'config': self.primary.get_config()})
        layer_configs.append({'class_name': 'DenseCaps', 'config': self.dense.get_config()})
        output_dict['encoder_config'] = {'layers': layer_configs}
        return output_dict

class MatrixCapsEncoder(keras.Model):
    '''The Matrix Capsule Network from the original paper on EM routing
    modified for fmri data
    '''

    def __init__(self, num_voxels):
        super(MatrixCapsEncoder, self).__init__()
        self.class_name = 'MatrixCapsEncoder'

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
        x, y = data

        with tf.GradientTape() as tape:
            # Propagate through capslayers
            filters = self.conv(x)
            pose1, a1 = self.primary(filters)
            pose2, a2 = self.convcaps1([pose1, a1])
            pose3, a3 = self.convcaps2([pose2, a2])
            pose4, a4 = self.classcaps([pose3, a3])
            ts = time.time()
            
            # Reconstruct image
            pose_shape = pose4.shape # [batch_size, num_caps, caps_dim[0], caps_dim[1]]
            decoder_input = tf.reshape(pose4, [-1, pose_shape[1] * pose_shape[2] * pose_shape[3]])
            y_pred = self.decoder(decoder_input)
            
            loss = self.compiled_loss(y, y_pred)

        # Calculate gradients
        training_vars = self.trainable_variables
        gradients = tape.gradient(loss, training_vars)

        # Optimize weights
        self.optimizer.apply_gradients(zip(gradients, training_vars))
        
        # Return loss and other metrics
        self.compiled_metrics.update_state(y, y_pred)
        output_dict = {m.name : m.result() for m in self.metrics}

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
        y_pred = self.decoder(decoder_input)

        loss = self.compiled_loss(y, y_pred)
        
        # Return loss and other metrics
        self.compiled_metrics.update_state(y, y_pred)
        output_dict = {m.name : m.result() for m in self.metrics}
        return output_dict

class ExperimentalEncoder(keras.Model):
    def __init__(self, num_voxels, num_output_capsules=32, routing='dynamic', caps_act='squash'):
        super(ExperimentalEncoder, self).__init__()
        self.class_name = 'ExperimentalEncoder'
        # Define layers
        self.conv = layers.Conv2D(filters=256, kernel_size=9, strides=(1, 1), padding='valid', activation='relu')
        self.primary = caps_layers.PrimaryCaps2D(num_channels=32, kernel_size=9, capsule_dim=8, strides=2, padding='valid', activation=caps_act)
        self.convcaps = caps_layers.ConvCaps2D(32, kernel_size=3, strides=1, capsule_dim=8, routing=routing, activation=caps_act)
        self.dense1 = caps_layers.DenseCaps(num_capsules=num_output_capsules, capsule_dim=16, routing=routing, activation='norm')

        self.decoder = keras.Sequential(
            [
                layers.Dense(512, activation='relu'),
                layers.Dense(1024, activation='relu'),
                layers.Dense(num_voxels)
            ],
            name='decoder'
        )
    
    def call(self, inputs): # Define this method solely so that we can build model and print model summary
        # Propagate through encoder
        conv_out = self.conv(inputs)
        pose, a = self.primary(conv_out)
        pose, a = self.convcaps([pose, a])
        pose, a  = self.dense1([pose, a])
        
        
        # get fmri_predictions
        pose_shape = pose.shape # [batch_size, num_caps, caps_dim[0], caps_dim[1]]
        decoder_input = tf.reshape(pose, [-1, pose_shape[1] * pose_shape[2] * pose_shape[3]])
        y_pred = self.decoder(decoder_input)

        return y_pred

    def train_step(self, data):
        x, y = data # x are input images, y is fmri voxels

        # Forward propagation
        ##############################
        with tf.GradientTape() as tape:
            # Propagate through encoder
            conv_out = self.conv(x)
            pose, a = self.primary(conv_out)
            pose, a = self.convcaps([pose, a])
            pose, a  = self.dense1([pose, a])
            
            
            # get fmri_predictions
            pose_shape = pose.shape # [batch_size, num_caps, caps_dim[0], caps_dim[1]]
            decoder_input = tf.reshape(pose, [-1, pose_shape[1] * pose_shape[2] * pose_shape[3]])
            y_pred = self.decoder(decoder_input)

            # Calculate loss
            loss = self.compiled_loss(y, y_pred)
        
        # Calculate gradients
        training_vars = self.trainable_variables
        gradients = tape.gradient(loss, training_vars)

        # Optimize weights
        self.optimizer.apply_gradients(zip(gradients, training_vars))

        # Return loss and other metrics
        self.compiled_metrics.update_state(y, y_pred)
        output_dict = {m.name : m.result() for m in self.metrics}
        return output_dict

    def test_step(self, data):
        x, y = data # x are input images, y are ohe labels

        # Propagate through encoder
        conv_out = self.conv(x)
        pose, a = self.primary(conv_out)
        pose, a = self.convcaps([pose, a])
        pose, a  = self.dense1([pose, a])
        
        
        # get fmri_predictions
        pose_shape = pose.shape # [batch_size, num_caps, caps_dim[0], caps_dim[1]]
        decoder_input = tf.reshape(pose, [-1, pose_shape[1] * pose_shape[2] * pose_shape[3]])
        y_pred = self.decoder(decoder_input)

        # Calculate loss
        loss = self.compiled_loss(y, y_pred)

        # Return loss and other metrics
        self.compiled_metrics.update_state(y, y_pred)
        output_dict = {m.name : m.result() for m in self.metrics}
        return output_dict

    def predict_step(self, inputs): # Define this method solely so that we can build model and print model summary
        # Propagate through encoder
        conv_out = self.conv(inputs)
        pose, a = self.primary(conv_out)
        pose, a = self.convcaps([pose, a])
        pose, a  = self.dense1([pose, a])
        
        # get fmri_predictions
        pose_shape = pose.shape # [batch_size, num_caps, caps_dim[0], caps_dim[1]]
        decoder_input = tf.reshape(pose, [-1, pose_shape[1] * pose_shape[2] * pose_shape[3]])
        y_pred = self.decoder(decoder_input)

        return y_pred

    def get_config(self):
        output_dict = {}
        layer_configs = []
        output_dict['decoder_config'] = self.decoder.get_config()
        layer_configs.append({'class_name': 'Conv2D', 'config': self.conv.get_config()})
        layer_configs.append({'class_name': 'PrimaryCaps2D', 'config': self.primary.get_config()})
        layer_configs.append({'class_name': 'ConvCaps2D', 'config': self.convcaps.get_config()})
        layer_configs.append({'class_name': 'DenseCaps', 'config': self.dense1.get_config()})
        output_dict['encoder_config'] = {'layers': layer_configs}

        return output_dict

class EncoderMach2(ExperimentalEncoder):
    '''
    Currently not working, init function of this class is not run for some reason
    '''
    def __init__(self, num_voxels, pretrained_weights_file='imagenet-caffe-ref.mat'):
        super(EncoderMach2, self).__init__(num_voxels=num_voxels)
        self.class_name = 'EncoderMach2'
        # Define layers
        pretrained_weights = scipy.io.loadmat(pretrained_weights_file)['layers']
        self.conv = layers.Lambda(tools.conv2d_relu_, 
            arguments={'net_layers': pretrained_weights, 
                        'layer': 0, 
                        'layer_name': 'conv1', 
                        'stride': 2,
                        'pad': 'SAME'}
                    )

        self.primary = caps_layers.PrimaryCaps2D(num_channels=32, kernel_size=9, capsule_dim=[4, 4], strides=2, padding='valid', activation='squash')
        self.convcaps = caps_layers.ConvCaps2D(32, kernel_size=3, strides=2, capsule_dim=[4,4], routing='EM')
        self.dense1 = caps_layers.DenseCaps(num_capsules=32, capsule_dim=[4, 4], routing='EM')

        self.decoder = keras.Sequential(
            [
                layers.Dense(512, activation='relu'),
                layers.Dense(1024, activation='relu'),
                layers.Dense(num_voxels)
            ],
            name='decoder'
        )

    def get_config(self):
        output_dict = {}
        layer_configs = []
        output_dict['decoder_config'] = self.decoder.get_config()
        layer_configs.append({'class_name': 'Conv2D', 'config': 'pretrained Conv Layer'})
        layer_configs.append({'class_name': 'PrimaryCaps2D', 'config': self.primary.get_config()})
        layer_configs.append({'class_name': 'ConvCaps2D', 'config': self.convcaps.get_config()})
        layer_configs.append({'class_name': 'DenseCaps', 'config': self.dense1.get_config()})
        output_dict['encoder_config'] = {'layers': layer_configs}

        return output_dict

class EncoderMach3(MatrixCapsEncoder):
    def __init__(self, num_voxels, pretrained_weights_file='imagenet-caffe-ref.mat'):
        super(EncoderMach3, self).__init__(num_voxels=num_voxels)
        self.class_name = 'EncoderMach3'
        # Define layers
        pretrained_weights = scipy.io.loadmat(pretrained_weights_file)['layers']
        self.conv = layers.Lambda(tools.conv2d_relu_, 
            arguments={'net_layers': pretrained_weights, 
                        'layer': 0, 
                        'layer_name': 'conv1', 
                        'stride': 2,
                        'pad': 'SAME'}
                    )

        self.primary = caps_layers.PrimaryCaps2D(num_channels=32, kernel_size=9, capsule_dim=[4, 4], strides=2, padding='valid', activation='sigmoid')
        self.convcaps1 = caps_layers.ConvCaps2D(32, kernel_size=3, strides=2, capsule_dim=[4,4], routing='EM')
        self.convcaps2 = caps_layers.ConvCaps2D(32, kernel_size=3, strides=2, capsule_dim=[4,4], routing='EM')
        self.classcaps = caps_layers.DenseCaps(num_capsules=32, capsule_dim=[4, 4], routing='EM')

        self.decoder = keras.Sequential(
            [
                layers.Dense(512, activation='relu'),
                layers.Dense(1024, activation='relu'),
                layers.Dense(num_voxels)
            ],
            name='decoder'
        )

    def get_config(self):
        output_dict = {}
        layer_configs = []
        output_dict['decoder_config'] = self.decoder.get_config()
        layer_configs.append({'class_name': 'Conv2D', 'config': 'pretrained Conv Layer'})
        layer_configs.append({'class_name': 'PrimaryCaps2D', 'config': self.primary.get_config()})
        layer_configs.append({'class_name': 'ConvCaps2D', 'config': self.convcaps1.get_config()})
        layer_configs.append({'class_name': 'ConvCaps2D', 'config': self.convcaps2.get_config()})
        layer_configs.append({'class_name': 'DenseCaps', 'config': self.classcaps.get_config()})
        output_dict['encoder_config'] = {'layers': layer_configs}

        return output_dict

class EncoderMach4(MatrixCapsEncoder):
    def __init__(self, num_voxels, pretrained_weights_file='imagenet-caffe-ref.mat'):
        super(EncoderMach4, self).__init__(num_voxels=num_voxels)
        self.class_name = 'EncoderMach4'
        # Define layers
        pretrained_weights = scipy.io.loadmat(pretrained_weights_file)['layers']
        self.conv = layers.Lambda(tools.conv2d_relu_, 
            arguments={'net_layers': pretrained_weights, 
                        'layer': 0, 
                        'layer_name': 'conv1', 
                        'stride': 2,
                        'pad': 'SAME'}
                    )

        self.primary = caps_layers.PrimaryCaps2D(num_channels=32, kernel_size=9, capsule_dim=[4, 4], strides=2, padding='valid', activation='sigmoid', kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=0.001))
        self.convcaps1 = caps_layers.ConvCaps2D(32, kernel_size=3, strides=2, capsule_dim=[4,4], routing='EM', kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=0.001))
        self.convcaps2 = caps_layers.ConvCaps2D(32, kernel_size=3, strides=2, capsule_dim=[4,4], routing='EM', kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=0.001))
        self.classcaps = caps_layers.DenseCaps(num_capsules=32, capsule_dim=[4, 4], routing='EM', regularizer=keras.regularizers.L1(l1=10))

        self.decoder = keras.Sequential(
            [
                layers.Dense(512, activation='relu'),
                layers.Dense(1024, activation='relu'),
                layers.Dense(num_voxels, kernel_regularizer=keras.regularizers.L1(l1=10))
            ],
            name='decoder'
        )

    def get_config(self):
        output_dict = {}
        layer_configs = []
        output_dict['decoder_config'] = self.decoder.get_config()
        layer_configs.append({'class_name': 'Conv2D', 'config': 'pretrained Conv Layer'})
        layer_configs.append({'class_name': 'PrimaryCaps2D', 'config': self.primary.get_config()})
        layer_configs.append({'class_name': 'ConvCaps2D', 'config': self.convcaps1.get_config()})
        layer_configs.append({'class_name': 'ConvCaps2D', 'config': self.convcaps2.get_config()})
        layer_configs.append({'class_name': 'DenseCaps', 'config': self.classcaps.get_config()})
        output_dict['encoder_config'] = {'layers': layer_configs}

        return output_dict

class EncoderMach5(keras.Model):
    def __init__(self, num_voxels, pretrained_weights_file='imagenet-caffe-ref.mat'):
        super(EncoderMach5, self).__init__()
        self.class_name = 'EncoderMach5'
        # Define layers
        pretrained_weights = scipy.io.loadmat(pretrained_weights_file)['layers'] # 11 by 11 input kernel
        self.conv = layers.Lambda(tools.conv2d_relu_, 
            arguments={'net_layers': pretrained_weights, 
                        'layer': 0, 
                        'layer_name': 'conv1', 
                        'stride': 2,
                        'pad': 'SAME'}
                    )
        self.norm1 = layers.BatchNormalization(axis=-1) # default is channels last
        self.primary = caps_layers.PrimaryCaps2D(num_channels=32, kernel_size=9, capsule_dim=[4, 4], strides=2, padding='valid', activation='sigmoid', kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=0.001))
        self.norm2 = layers.BatchNormalization(axis=-3) # Channels is third last dim (two dims for capsules)
        self.convcaps1 = caps_layers.ConvCaps2D(32, kernel_size=3, strides=2, capsule_dim=[4,4], routing='EM', kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=0.001))
        self.norm3 = layers.BatchNormalization(axis=-3) # Channels is third last dim (two dims for capsules)
        self.convcaps2 = caps_layers.ConvCaps2D(32, kernel_size=3, strides=2, capsule_dim=[4,4], routing='EM', kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=0.001))
        self.norm4 = layers.BatchNormalization(axis=-3) # Channels is third last dim (two dims for capsules)
        self.densecaps = caps_layers.DenseCaps(num_capsules=32, capsule_dim=[4, 4], routing='EM', regularizer=keras.regularizers.L1(l1=10))

        self.decoder = keras.Sequential(
            [
                layers.Dense(512, activation='relu'),
                layers.Dense(1024, activation='relu'),
                layers.Dense(num_voxels, kernel_regularizer=keras.regularizers.L1(l1=10))
            ],
            name='decoder'
        )
    
    def call(self, inputs):
        # Propagate through encoder
        conv_out = self.conv(inputs)
        conv_norm = self.norm1(conv_out)
        pose, a = self.primary(conv_norm)
        pose_norm = self.norm2(pose)
        pose, a = self.convcaps1([pose_norm, a])
        pose_norm = self.norm3(pose)
        pose, a = self.convcaps2([pose_norm, a])
        pose_norm = self.norm4(pose)
        pose, a = self.densecaps([pose_norm, a])

        # Construct fmri predictions from capsules
        pose_shape = pose.shape
        decoder_input = tf.reshape(pose, [-1, pose_shape[1] * pose_shape[2] * pose_shape[3]])
        y_pred = self.decoder(decoder_input)

        return y_pred

    def train_step(self, data):
        x, y = data # x are input images, y is fmri voxels

        # Forward propagation
        ##############################
        with tf.GradientTape() as tape:
            # Propagate through encoder
            conv_out = self.conv(x)
            conv_norm = self.norm1(conv_out)
            pose, a = self.primary(conv_norm)
            pose_norm = self.norm2(pose)
            pose, a = self.convcaps1([pose_norm, a])
            pose_norm = self.norm3(pose)
            pose, a = self.convcaps2([pose_norm, a])
            pose_norm = self.norm4(pose)
            pose, a = self.densecaps([pose_norm, a])
            
            # Construct fmri predictions from capsules
            pose_shape = pose.shape
            decoder_input = tf.reshape(pose, [-1, pose_shape[1] * pose_shape[2] * pose_shape[3]])
            y_pred = self.decoder(decoder_input)

            # Calculate loss
            loss = self.compiled_loss(y, y_pred)
        
        # Calculate gradients
        training_vars = self.trainable_variables
        gradients = tape.gradient(loss, training_vars)

        # Optimize weights
        self.optimizer.apply_gradients(zip(gradients, training_vars))

        # Return loss and other metrics
        self.compiled_metrics.update_state(y, y_pred)
        output_dict = {m.name : m.result() for m in self.metrics}
        return output_dict

    def test_step(self, data):
        x, y = data # x are input images, y is fmri voxels

        # Propagate through encoder
        conv_out = self.conv(x)
        conv_norm = self.norm1(conv_out)
        pose, a = self.primary(conv_norm)
        pose_norm = self.norm2(pose)
        pose, a = self.convcaps1([pose_norm, a])
        pose_norm = self.norm3(pose)
        pose, a = self.convcaps2([pose_norm, a])
        pose_norm = self.norm4(pose)
        pose, a = self.densecaps([pose_norm, a])
        
        # Construct fmri predictions from capsules
        pose_shape = pose.shape
        decoder_input = tf.reshape(pose, [-1, pose_shape[1] * pose_shape[2] * pose_shape[3]])
        y_pred = self.decoder(decoder_input)

        # Calculate loss
        loss = self.compiled_loss(y, y_pred)

        # Return loss and other metrics
        self.compiled_metrics.update_state(y, y_pred)
        output_dict = {m.name : m.result() for m in self.metrics}
        return output_dict

    def get_config(self):
        output_dict = {}
        layer_configs = []
        output_dict['decoder_config'] = self.decoder.get_config()
        layer_configs.append({'class_name': 'Conv2D', 'config': 'pretrained Conv Layer'})
        layer_configs.append({'class_name': 'BatchNormalization', 'config': self.norm1.get_config()})
        layer_configs.append({'class_name': 'PrimaryCaps2D', 'config': self.primary.get_config()})
        layer_configs.append({'class_name': 'BatchNormalization', 'config': self.norm2.get_config()})
        layer_configs.append({'class_name': 'ConvCaps2D', 'config': self.convcaps1.get_config()})
        layer_configs.append({'class_name': 'BatchNormalization', 'config': self.norm3.get_config()})
        layer_configs.append({'class_name': 'ConvCaps2D', 'config': self.convcaps2.get_config()})
        layer_configs.append({'class_name': 'BatchNormalization', 'config': self.norm4.get_config()})
        layer_configs.append({'class_name': 'DenseCaps', 'config': self.densecaps.get_config()})
        output_dict['encoder_config'] = {'layers': layer_configs}

        return output_dict

class EncoderMach6(EncoderMach5):
    def __init__(self, num_voxels, pretrained_weights_file='imagenet-caffe-ref.mat'):
        super(EncoderMach6, self).__init__(num_voxels, pretrained_weights_file)
        self.class_name = 'EncoderMach6'
        # Define layers
        self.conv = layers.Conv2D(filters=256, kernel_size=(9, 9), strides=(2, 2), padding='valid', activation='relu')
        self.norm1 = layers.BatchNormalization(axis=-1) # default is channels last
        self.primary = caps_layers.PrimaryCaps2D(num_channels=32, kernel_size=9, capsule_dim=[4, 4], strides=2, padding='valid', activation='sigmoid', kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=0.001))
        self.norm2 = layers.BatchNormalization(axis=-3) # Channels is third last dim (two dims for capsules)
        self.convcaps1 = caps_layers.ConvCaps2D(32, kernel_size=3, strides=2, capsule_dim=[4,4], routing='EM', kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=0.001))
        self.norm3 = layers.BatchNormalization(axis=-3) # Channels is third last dim (two dims for capsules)
        self.convcaps2 = caps_layers.ConvCaps2D(32, kernel_size=3, strides=2, capsule_dim=[4,4], routing='EM', kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=0.001))
        self.norm4 = layers.BatchNormalization(axis=-3) # Channels is third last dim (two dims for capsules)
        self.densecaps = caps_layers.DenseCaps(num_capsules=32, capsule_dim=[4, 4], routing='EM', regularizer=keras.regularizers.L1(l1=10))

        self.decoder = keras.Sequential(
            [
                layers.Dense(32, activation='relu'),
                layers.Dense(128, activation='relu'),
                layers.Dense(256, activation='relu'),
                layers.Dense(100, activation='relu'),
                layers.Dense(num_voxels, kernel_regularizer=keras.regularizers.L1(l1=10))
            ],
            name='decoder'
        )

    def get_config(self):
        output_dict = {}
        layer_configs = []
        output_dict['decoder_config'] = self.decoder.get_config()
        layer_configs.append({'class_name': 'Conv2D', 'config': self.conv.get_config()})
        layer_configs.append({'class_name': 'BatchNormalization', 'config': self.norm1.get_config()})
        layer_configs.append({'class_name': 'PrimaryCaps2D', 'config': self.primary.get_config()})
        layer_configs.append({'class_name': 'BatchNormalization', 'config': self.norm2.get_config()})
        layer_configs.append({'class_name': 'ConvCaps2D', 'config': self.convcaps1.get_config()})
        layer_configs.append({'class_name': 'BatchNormalization', 'config': self.norm3.get_config()})
        layer_configs.append({'class_name': 'ConvCaps2D', 'config': self.convcaps2.get_config()})
        layer_configs.append({'class_name': 'BatchNormalization', 'config': self.norm4.get_config()})
        layer_configs.append({'class_name': 'DenseCaps', 'config': self.densecaps.get_config()})
        output_dict['encoder_config'] = {'layers': layer_configs}

        return output_dict

class EncoderMach7(ExperimentalEncoder):
    def __init__(self, num_voxels, routing='EM', caps_act='sigmoid'):
        super(EncoderMach7, self).__init__(num_voxels)
        self.class_name = 'EncoderMach7'
        # Define layers
        self.conv = layers.Conv2D(filters=32, kernel_size=9, strides=(2, 2), padding='valid', activation='relu')
        self.primary = caps_layers.PrimaryCaps2D(num_channels=16, kernel_size=9, capsule_dim=[4, 4], strides=2, padding='valid', activation=caps_act)
        self.convcaps = caps_layers.ConvCaps2D(16, kernel_size=3, strides=2, capsule_dim=[4,4], routing=routing, routing_iterations=3, activation=caps_act, padding='same')
        self.dense1 = caps_layers.DenseCaps(num_capsules=16, capsule_dim=[4, 4], routing=routing, routing_iterations=3, activation='norm')

        self.decoder = keras.Sequential(
            [
                layers.Dense(1024, activation='relu'),
                layers.Dense(num_voxels)
            ],
            name='decoder'
        )

class EncoderMach7wide(ExperimentalEncoder):
    def __init__(self, num_voxels, routing='EM', caps_act='sigmoid'):
        super(EncoderMach7wide, self).__init__(num_voxels)
        self.class_name = 'EncoderMach7wide'
        # Define layers
        self.conv = layers.Conv2D(filters=128, kernel_size=9, strides=(2, 2), padding='valid', activation='relu')
        self.primary = caps_layers.PrimaryCaps2D(num_channels=32, kernel_size=9, capsule_dim=[4, 4], strides=2, padding='valid', activation=caps_act)
        self.convcaps = caps_layers.ConvCaps2D(32, kernel_size=3, strides=2, capsule_dim=[4,4], routing=routing, routing_iterations=3, activation=caps_act)
        self.dense1 = caps_layers.DenseCaps(num_capsules=32, capsule_dim=[4, 4], routing=routing, routing_iterations=3)

        self.decoder = keras.Sequential(
            [
                layers.Dense(512, activation='relu'),
                layers.Dense(1024, activation='relu'),
                layers.Dense(num_voxels)
            ],
            name='decoder'
        )

class EncoderMach8(ExperimentalEncoder):
    def __init__(self, num_voxels, routing='dynamic', caps_act='squash'):
        super(EncoderMach8, self).__init__(num_voxels)
        self.class_name = 'EncoderMach7wide'
        # Define layers
        self.conv = layers.Conv2D(filters=256, kernel_size=9, strides=(1, 1), padding='valid', activation='relu')
        self.primary = caps_layers.PrimaryCaps2D(num_channels=32, kernel_size=9, capsule_dim=8, strides=2, padding='valid', activation=caps_act)
        self.convcaps = caps_layers.ConvCaps2D(32, kernel_size=3, strides=1, capsule_dim=8, routing=routing, activation=caps_act, padding='valid')
        self.dense1 = caps_layers.DenseCaps(num_capsules=2, capsule_dim=16, routing=routing, activation=caps_act)

        self.decoder = keras.Sequential(
            [
                layers.Dense(512, activation='relu'),
                layers.Dense(1024, activation='relu'),
                layers.Dense(num_voxels)
            ],
            name='decoder'
        )

class EncoderMach9(EncoderMach7):
    def call(self, inputs):
        # Propagate through encoder
        conv_out = self.conv(inputs)
        pose, a = self.primary(conv_out)
        pose, a = self.convcaps([pose, a])
        pose, a  = self.dense1([pose, a])
        
        
        # get fmri_predictions
        pose_shape = pose.shape # [batch_size, num_caps, caps_dim[0], caps_dim[1]]
        poses_flat = tf.reshape(pose, [-1, pose_shape[1] * pose_shape[2] * pose_shape[3]])
        decoder_input = tf.concat([poses_flat, a], axis=-1)
        y_pred = self.decoder(decoder_input)

        return y_pred

class EncoderMach10(CapsEncoder):
    # Add regularization
    def __init__(self, num_voxels, output_activation='linear', num_output_capsules=10, routing='dynamic', caps_act='squash'):
        super(CapsEncoder, self).__init__()
        self.class_name = 'CapsEncoder'

        # Define layers
        self.conv = layers.Conv2D(filters=256, kernel_size=9, strides=(1, 1), padding='valid', activation='relu', kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=0.001))
        self.primary = caps_layers.PrimaryCaps2D(num_channels=32, kernel_size=9, capsule_dim=8, strides=2, padding='valid', activation=caps_act, kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=0.001))
        self.dense = caps_layers.DenseCaps(num_capsules=num_output_capsules, capsule_dim=16, routing=routing, activation='squash', kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=0.001))

        self.decoder = keras.Sequential(
            [
                layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=0.001)),
                layers.Dense(1024, activation='relu', kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=0.001)),
                layers.Dense(num_voxels, activation=output_activation, kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=0.001))
            ],
            name='decoder'
        )

class EncoderMach11(CapsEncoder):
    def __init__(self, num_voxels, output_activation='linear', num_output_capsules=10, routing='dynamic', caps_act='squash'):
        super(CapsEncoder, self).__init__()
        self.class_name = 'CapsEncoder'

        # Define layers
        self.conv = layers.Conv2D(filters=128, kernel_size=9, strides=(1, 1), padding='valid', activation='relu')
        self.primary = caps_layers.PrimaryCaps2D(num_channels=8, kernel_size=9, capsule_dim=8, strides=2, padding='valid', activation=caps_act)
        self.dense = caps_layers.DenseCaps(num_capsules=num_output_capsules, capsule_dim=16, routing=routing, activation='squash', name='class_capsules')

        self.decoder = keras.Sequential(
            [
                layers.Dense(512, activation='relu'),
                layers.Dense(1024, activation='relu'),
                layers.Dense(num_voxels, activation=output_activation)
            ],
            name='decoder'
        )

    