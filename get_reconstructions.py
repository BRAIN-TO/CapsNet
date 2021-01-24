#imports
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from models import CapsNet
import losses
import numpy as np


print('Importing Data...')
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')/255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')/255.0

print('Loading Model...')
model = keras.models.load_model('models/CapsNet_mnist/saved_model', custom_objects={'margin_recon_loss' : losses.margin_recon_loss})

print('Getting Reconstructions')
sample = x_train[100, :, :, :]
sample = tf.expand_dims(sample, axis=0) # Add dim for batch size back
print(tf.shape(sample))
a, p, og_image = model.call(sample)

og_image = tf.reshape(og_image, [28, 28, 1])
tf.keras.preprocessing.image.save_img('images/im100.png', og_image, data_format='channels_last')

im2 = model.reconstruct_image(a, p)
im2 = tf.reshape(im2, [28, 28, 1])
tf.keras.preprocessing.image.save_img('images/im100-v2.png', im2, data_format='channels_last')

# Modify capsules to see how dimensions affect reconstruction
dim = [5, 0] # Dimensions to vary
p = tf.tile(p, [11, 1, 1, 1]).numpy() # shape: [batch_size, num_capsules, caps_dim[0], caps_dim[1]]
a = tf.tile(a, [11, 1])

vals = np.linspace(-0.5, 0.5, num=11) # Get values to vary dimension by
print(tf.shape(p))
#Add vals to dim of capsule
for i, val in enumerate(vals):
    # Add for every capsule since all but one capsule will be masked/zeroed anyways
    p[i, :, dim[0], dim[1]] = p[i, :, dim[0], dim[1]] + val 

images = model.reconstruct_image(a, p)
print(tf.shape(images))
collage = tf.reshape(images, [11*28, 28, 1])
tf.keras.preprocessing.image.save_img('images/collage.png', collage, data_format='channels_last')