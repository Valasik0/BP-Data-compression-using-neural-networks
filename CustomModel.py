import tensorflow as tf
from keras.layers import LSTM, Dense, Flatten, LeakyReLU

class CustomModel(tf.keras.Model):
    def __init__(self, layers, k, sigma):
        super(CustomModel, self).__init__()

        self.layers_list = []
        self.in_shape = (k, sigma)

        for i, layer in enumerate(layers):
            layer_type, neurons, activation = layer
            if activation == 'LeakyReLU':
                activation = LeakyReLU(alpha=0.01)
            if layer_type == "Dense":
                self.layers_list.append(Dense(units=int(neurons), activation=activation))
            elif layer_type == "LSTM":
                if i < len(layers) - 1 and layers[i + 1][0] == "LSTM":
                    return_sequences = True
                else:
                    return_sequences = False
                if i == 0:
                    # If it's the first layer, set the input_shape
                    self.layers_list.append(LSTM(units=int(neurons), activation=activation, return_sequences=return_sequences, input_shape=self.in_shape))
                else:
                    self.layers_list.append(LSTM(units=int(neurons), activation=activation, return_sequences=return_sequences))
            elif layer_type == "Flatten":
                self.layers_list.append(Flatten())

        # output layer
        self.output_layer = Dense(units=sigma, activation='softmax')

    def call(self, inputs):
        x = inputs
        for layer in self.layers_list:
            x = layer(x)
        return self.output_layer(x)

