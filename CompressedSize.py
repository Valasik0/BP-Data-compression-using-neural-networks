import numpy as np
import tensorflow as tf
from TextAnalyzer import *
import tkinter as tk

class CompressedSize(TextAnalyzer):
    def __init__(self, model, k, batch_size, text):
        super().__init__(text)
        self.model = model
        self.k = k
        self.batch_size = batch_size
        self.epsilon = 1e-10
        self.compressed_size = 0

    def compute(self, text):
        
        try:
            self.model.summary()
        except AttributeError:
            tk.messagebox.showerror("Error", "Model error")
            return None

        batch_sequences = []
        batch_chars = []
        n = len(text)

        # předzpracování textu
        mapped_text = [self.mapped_chars[char] if char in self.mapped_chars else self.mapped_chars['<UNK>'] for char in text]

        for i in range(self.k, len(mapped_text)):
            sequence = mapped_text[i - self.k:i]
            batch_sequences.append(tf.keras.utils.to_categorical(sequence, num_classes=self.sigma))
            actual_char_index = mapped_text[i]
            batch_chars.append(actual_char_index)

            if len(batch_sequences) == self.batch_size or i == len(mapped_text) - 1:
                probabilities = self.model.predict(np.array(batch_sequences))  #všechny pravděpodobnosti pro následující znak
                p_x = probabilities[np.arange(len(batch_sequences)), batch_chars]  # pravděpodobnosti skutečných následujících znaků
                self.compressed_size -= np.sum(np.log2(p_x + self.epsilon))

                batch_sequences = []
                batch_chars = []
        return self.compressed_size