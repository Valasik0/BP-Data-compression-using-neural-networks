import tensorflow as tf
from keras import layers

class CustomModel(tf.keras.Model):
    def __init__(self, io_size, k, num_dense_layers, dense_layer_sizes, 
                 num_lstm_layers, lstm_layer_sizes):
        super(CustomModel, self).__init__()

        self.num_lstm_layers = num_lstm_layers
        self.num_dense_layers = num_dense_layers
        self.dense_layer_sizes = dense_layer_sizes
        self.lstm_layer_sizes = lstm_layer_sizes
        self.io_size = io_size
        self.k = k
        
        if self.num_lstm_layers > 0:
            # input layer
            self.input_lstm = tf.keras.layers.LSTM(64, return_sequences=(self.num_lstm_layers > 0), 
                                                   input_shape=(self.k, self.io_size))

            self.lstm_layers = [tf.keras.layers.LSTM(units=size, 
                                                    return_sequences=(False if i == self.num_lstm_layers - 1 else True)) 
                                                    for i, size in enumerate(self.lstm_layer_sizes)]
            self.flatten_layer = tf.keras.layers.Flatten()

        else:
            # input layer
            self.flatten_layer = tf.keras.layers.Flatten(input_shape=(self.k, self.io_size))

            self.dense_layer = tf.keras.layers.Dense(64, activation='relu')

        

        self.dense_layers = [tf.keras.layers.Dense(units=size, activation='relu') 
                             for size in self.dense_layer_sizes]

        # output layer
        self.output_layer = tf.keras.layers.Dense(units=self.io_size, activation='softmax')

    def call(self, inputs):
        # If the model has LSTM layers
        if self.num_lstm_layers > 0:
            # Pass the input through the LSTM layers
            x = self.input_lstm(inputs)
            for layer in self.lstm_layers:
                x = layer(x)
            # Flatten the output of the LSTM layers for dense layers
            x = self.flatten_layer(x)
        else:
            # If the model doesn't have LSTM layers flatten the input and pass it through the Dense layer
            x = self.flatten_layer(inputs)
            x = self.dense_layer(x)

        # Pass the output through the Dense layers
        for layer in self.dense_layers:
            x = layer(x)
        # Pass the output through the output layer
        return self.output_layer(x)

