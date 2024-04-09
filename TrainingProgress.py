import tensorflow as tf
import time

class TrainingProgress(tf.keras.callbacks.Callback):
    def __init__(self, text_widget, total_epochs):
        super().__init__()
        self.text_widget = text_widget
        self.total_epochs = total_epochs
        self.last_update = time.time()
        self.logs = {}
        self.current_epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch + 1
        self.text_widget.insert('end', "\n")  # Insert a new line at the beginning of each epoch

    def on_train_batch_end(self, batch, logs=None):
        self.logs = logs or {}
        if time.time() - self.last_update > 0.1:
            text = "Epochs: {}/{}, ETA: {}, Loss: {}, Accuracy: {}".format(
                self.current_epoch,
                self.total_epochs,
                self.logs.get('eta'),
                self.logs.get('loss'),
                self.logs.get('accuracy')
            )

            # Get the current line and replace it with the new text
            current_line_index = self.text_widget.index('end-2c').split('.')[0]
            self.text_widget.delete(f'{current_line_index}.0', 'end')
            self.text_widget.insert(f'{current_line_index}.0', text)
            self.last_update = time.time()