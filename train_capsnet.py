#Public API's
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
# Custom Imports
from models import CapsNet
import losses
import json

'''Experimented with using callbacks to save a model at various checkpoints
throughout training (eg. every 10 epochs).
    -However for some reason callback only saves weights and not entire model
'''

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')/255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')/255.0
model_name = 'CapsNet_mnist_2'

print('Train Set: ', tf.shape(x_train))
print('Test Set: ', tf.shape(x_test))

# One hot encode the labels
y_train = tf.one_hot(y_train, depth=10)
y_test = tf.one_hot(y_test, depth=10)

training_params = {} # Dictionary to keep track of params so they can be saved
training_params['batch_size'] = 100
training_params['val_batch_size'] = 100
training_params['epochs'] = 50
training_params['loss'] = 'margin_recon_loss'
optimizer = keras.optimizers.Adam()
training_params['optimizer'] = optimizer.get_config()

# Create callback for saving model at various checkpoints
model_checkpoint = keras.callbacks.ModelCheckpoint(
    'models/CapsNet-mnist/saved_model_{epoch:02d}',
    save_weights_only=False,
    monitor='accuracy',
    save_best_only=False,
    save_freq='epoch',
    period=5
)

# Create distributed learning strategy object
strategy = tf.distribute.MirroredStrategy()
print("Number of devices in use: {}".format(strategy.num_replicas_in_sync))
acc_metric = keras.metrics.SparseCategoricalAccuracy(name='accuracy')

with strategy.scope():
    model = CapsNet()
    model.compile(
        optimizer=optimizer,
        loss=losses.margin_recon_loss, # reminder to not include parentheses here
        metrics=['accuracy']
    )
    model.build(x_train.shape)

print(model.summary())
#model._set_inputs(x_train)

training = model.fit(x=x_train, y=y_train, batch_size=training_params['batch_size'], epochs=training_params['epochs'], verbose=1, validation_data=(x_test, y_test), callbacks=[model_checkpoint])

# Save model and model history

#model.save('models/CapsNet_mnist/saved_model', save_format='tf')

testing = model.evaluate(x_test, y_test, batch_size=training_params['val_batch_size'], return_dict=True)

# Model Saving
if save:
    # Save full model. Also save weights
    model.save_weights('models/' + model_name + '/model_weights', save_format='tf')

    with open('models/' + model_name + '/train_config.yaml', 'w') as file:
        yaml.dump(training_params, file) # Save archetecture

    # with open('models/' + model_name + '/config.yaml', 'w') as file:
    #     yaml.dump(model.get_config(), file) # Save archetecture

    with open('models/' + model_name + '/train-history.json', 'w') as file:
        # df = pd.DataFrame(training.history)
        # df.to_json(file)
        json.dump(training.history, file) # save training history

    with open('models/' + model_name + '/test-history.json', 'w') as file:
        # df = pd.DataFrame(testing)
        # df.to_json(file)
        json.dump(testing, file) # save testing history