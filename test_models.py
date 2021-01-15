#imports
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from models import CapsNet
import losses

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')/255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')/255.0

# One hot encode the labels
y_train = tf.one_hot(y_train, depth=10)
y_test = tf.one_hot(y_test, depth=10)
#print(tf.shape(y_train))

acc_metric = keras.metrics.SparseCategoricalAccuracy(name='accuracy')
model = CapsNet()
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=losses.margin_recon_loss, # reminder to not include parentheses here
    metrics=['accuracy']
)

model.fit(x=x_train, y=y_train, batch_size=128, epochs=2, verbose=1)
print(model.summary())

