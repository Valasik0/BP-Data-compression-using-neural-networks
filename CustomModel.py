import tensorflow as tf
from keras import layers

class CustomModel(tf.keras.Model):
    def __init__(self, io_size=128, k=3, num_dense_layers=3, dense_layer_sizes=None, dense_activations=None):
        super(CustomModel, self).__init__()

        self.lstm1 = layers.LSTM(64, return_sequences=True, input_shape=(k, io_size)) #input layer
        self.lstm2 = layers.LSTM(64)

        if dense_layer_sizes is None:
            dense_layer_sizes = [50, 100, 150]

        if dense_activations is None:
            dense_activations = ['relu'] * num_dense_layers

        self.dense_layers = [layers.Dense(size, activation=activation) for size, activation in zip(dense_layer_sizes, dense_activations)]
        self.dropout = layers.Dropout(0.2)
        self.output_layer = layers.Dense(io_size, activation='softmax') #output layer

    def call(self, inputs):
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        for layer in self.dense_layers:
            x = layer(x)
        x = self.dropout(x)
        return self.output_layer(x)

