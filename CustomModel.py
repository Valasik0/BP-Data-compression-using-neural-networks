import tensorflow as tf
from tensorflow import keras
from keras import layers

class CustomModel(tf.keras.Model):
    def __init__(self, io_size=128, k=3):
        super(CustomModel, self).__init__()

        self.lstm = layers.LSTM(50, input_shape=(k, io_size))
        self.dense_layers = [
            layers.Dense(50, activation='relu'),
            layers.Dense(100, activation='relu'),
            layers.Dense(150, activation='relu')
        ]
        self.output_layer = layers.Dense(io_size, activation='softmax')

    def call(self, inputs):
        x = self.lstm(inputs)
        for layer in self.dense_layers:
            x = layer(x)
        return self.output_layer(x)