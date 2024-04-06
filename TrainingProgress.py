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

    def on_train_batch_end(self, batch, logs=None):
        self.logs = logs or {}
        if time.time() - self.last_update > 1:
            text = "Epochs: {}/{}, ETA: {}, Loss: {}, Accuracy: {}\n".format(
                self.current_epoch,
                self.total_epochs,
                self.logs.get('eta'),
                self.logs.get('loss'),
                self.logs.get('accuracy')
            )

            self.text_widget.delete('1.0', 'end')
            self.text_widget.insert('end', text)
            self.last_update = time.time()