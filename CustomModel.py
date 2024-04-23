import tensorflow as tf
from keras.layers import LSTM, Dense, Flatten, LeakyReLU
from keras.models import Sequential

class CustomModel(tf.keras.Model):
    def __init__(self, layers, k, sigma):
        super(CustomModel, self).__init__()

        self.model = Sequential()
        self.in_shape = (k, sigma)

        for i, layer in enumerate(layers):
            layer_type, neurons, activation = layer
            if activation == 'LeakyReLU':
                activation = LeakyReLU(alpha=0.01)
            if layer_type == "Dense":
                if i == 0:
                    self.model.add(Flatten(input_shape=self.in_shape))
                self.model.add(Dense(units=int(neurons), activation=activation))
            elif layer_type == "LSTM":
                if i < len(layers) - 1 and layers[i + 1][0] == "LSTM":
                    return_sequences = True
                else:
                    return_sequences = False
                if i == 0:
                    self.model.add(LSTM(units=int(neurons), activation=activation, return_sequences=return_sequences, input_shape=self.in_shape))
                else:
                    self.model.add(LSTM(units=int(neurons), activation=activation, return_sequences=return_sequences))
            elif layer_type == "Flatten":
                self.model.add(Flatten())

        # output layer
        self.model.add(Dense(units=sigma, activation='softmax'))

    def call(self, inputs):
        return self.model(inputs)

