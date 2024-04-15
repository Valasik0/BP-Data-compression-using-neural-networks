import tensorflow as tf
import numpy as np

class SequencesGenerator:
    def __init__(self, text, k, mapped_chars, sigma, batch_size):
        self.text = text
        self.k = k
        self.mapped_chars = mapped_chars
        self.sigma = sigma
        self.batch_size = batch_size

    def generate_sequences(self):
        while True:
            batch_chars = []
            batch_labels = []

            for i in range(self.k, len(self.text)):
                sequence = self.text[i - self.k:i + 1]  # k = 3, text = abrakadabra, abra, brak, raka, akad .....
                mapped_sequence = [self.mapped_chars[char] for char in sequence] # [1,2,3,1], [2,3,1,4], [3,1,4,1], [1,4,1,5] ... namapuje znaky na cisla
                train_chars = tf.keras.utils.to_categorical(mapped_sequence[:-1], num_classes=self.sigma)
                train_label = tf.keras.utils.to_categorical(mapped_sequence[-1], num_classes=self.sigma)
                batch_chars.append(train_chars)
                batch_labels.append(train_label)

                if len(batch_chars) == self.batch_size:
                    yield np.array(batch_chars), np.array(batch_labels)
                    batch_chars = []
                    batch_labels = []

            if batch_chars:  # yield zbývajících znaků na konci dávky
                yield np.array(batch_chars), np.array(batch_labels)