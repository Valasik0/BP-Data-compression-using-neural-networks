import tensorflow as tf
from keras import layers

class CustomModel(tf.keras.Model):
    def __init__(self, io_size, k, num_dense_layers, dense_layer_sizes, 
                 num_lstm_layers, lstm_layer_sizes):
        super(CustomModel, self).__init__()

        # input layer
        self.input_lstm = tf.keras.layers.LSTM(64, return_sequences=(num_lstm_layers > 0), input_shape=(k, io_size))

        self.lstm_layers = [tf.keras.layers.LSTM(units=size, 
                                                 return_sequences=(False if i == num_lstm_layers - 1 else True)) 
                                                 for i, size in enumerate(lstm_layer_sizes)]

        self.flatten_layer = tf.keras.layers.Flatten()

        self.dense_layers = [tf.keras.layers.Dense(units=size, activation='relu') 
                             for size in dense_layer_sizes]

        self.output_layer = tf.keras.layers.Dense(units=io_size, activation='softmax')

    def call(self, inputs):
        x = self.input_lstm(inputs)
        for layer in self.lstm_layers:
            x = layer(x)
        x = self.flatten_layer(x)
        for layer in self.dense_layers:
            x = layer(x)
        return self.output_layer(x)

