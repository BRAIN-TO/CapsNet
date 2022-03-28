print('Importing Packages...')
#Public API's
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
# Custom Imports
from models import CapsNet, MatrixCapsNet, HybridCapsNet, CapsRecon
import losses
import json

print('Loading Data...')
save = False # Whether or not to save the model
model_name = 'LowMem_mnist' # Model name

# Load MNIST Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')/255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')/255.0

# One hot encode the labels
y_train = tf.one_hot(y_train, depth=10)
y_test = tf.one_hot(y_test, depth=10)

print('Building Model...')
# Accuracy metric for classification models
acc_metric = keras.metrics.SparseCategoricalAccuracy(name='accuracy')

# Create distributed learning strategy object
strategy = tf.distribute.MirroredStrategy()
print("Number of GPU's in use: {}".format(strategy.num_replicas_in_sync))

with strategy.scope():
    model = MatrixCapsNet()
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=losses.spread_loss, # reminder to not include parentheses here
        metrics=['accuracy']
    )

    model.build(x_train.shape) # build model so that we can print summary

# Print model summary
print(model.summary())

# Train and Test Model
training = model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), batch_size=20, epochs=10, verbose=1)
testing = model.evaluate(x_test, y_test, batch_size=20, return_dict=True)

# Model Saving
if save:
    # Save model and model history
    model._set_inputs(x_train)
    model.save('models/' + model_name + '/saved_model', save_format='tf') # Saves whole model

    with open('models/' + model_name + '/train-history.json', 'w') as file:
        json.dump(training.history, file) # save training history

    with open('models/' + model_name + '/test-history.json', 'w') as file:
        json.dump(testing, file) # save testing history