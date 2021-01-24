#imports
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from models import CapsNet
import losses
import json

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')/255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')/255.0

print('Train Set: ', tf.shape(x_train))
print('Test Set: ', tf.shape(x_test))

# One hot encode the labels
y_train = tf.one_hot(y_train, depth=10)
y_test = tf.one_hot(y_test, depth=10)

acc_metric = keras.metrics.SparseCategoricalAccuracy(name='accuracy')
model = CapsNet()
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=losses.margin_recon_loss, # reminder to not include parentheses here
    metrics=['accuracy']
)

model.build(x_train.shape)
print(model.summary())

training = model.fit(x=x_train, y=y_train, batch_size=100, epochs=50, verbose=1)

# Save model and model history
model._set_inputs(x_train)
model.save('models/CapsNet_mnist/saved_model', save_format='tf')

with open('models/CapsNet_mnist/train-history.json', 'w') as file:
    json.dump(training.history, file)

testing = model.evaluate(x_test, y_test, batch_size=100, return_dict=True)

with open('models/CapsNet_mnist/test-history.json', 'w') as file:
    json.dump(testing, file)