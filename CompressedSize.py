import numpy as np
import tensorflow as tf
from TextAnalyzer import *
import tkinter as tk
import time

class CompressedSize(TextAnalyzer):
    def __init__(self, model, k, batch_size, text, text_widget):
        super().__init__(text)
        self.model = model
        self.k = k
        self.batch_size = batch_size
        self.epsilon = 1e-10
        self.compressed_size = 0
        self.text_widget = text_widget
        self.last_update = time.time()

    def compute(self, text):
        
        try:
            self.model.summary()
        except AttributeError:
            tk.messagebox.showerror("Error", "Model error")
            return None
        
        model_input_context = self.model.layers[0].input_shape[1]
        model_input_sigma = self.model.layers[0].input_shape[2]

        if model_input_context != self.k:
            tk.messagebox.showerror("Error", f"Model input shape ({model_input_context}) doesn't match selected context length ({self.k})")
            return None
        
        if model_input_sigma != self.sigma:
            tk.messagebox.showerror("Error", f"Model input alphabet ({model_input_sigma}) doesn't match text alphabet ({self.sigma})")
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

                if self.text_widget.winfo_exists():
                    if time.time() - self.last_update > 1.0:
                        self.text_widget.delete('1.0', 'end')
                        self.text_widget.insert('end', f"Compressed size: {round(self.compressed_size, 3)} bits\n")
                        self.last_update = time.time()

        return self.compressed_size