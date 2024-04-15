import tensorflow as tf
import time

class TrainingProgress(tf.keras.callbacks.Callback):
    def __init__(self, text_widget, total_epochs):
        super().__init__()
        self.text_widget = text_widget
        self.total_epochs = total_epochs
        self.start_time = time.time()
        self.last_update = time.time()
        self.logs = {}
        self.current_epoch = 0
        self.all_text = ""
        self.tmp_text = ""

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch + 1
    
    def on_epoch_end(self, epoch, logs=None):
        self.all_text = self.all_text + '\n'
        self.tmp_text = self.all_text
    
    def on_train_batch_end(self, batch, logs=None):
        self.logs = logs or {}
        if time.time() - self.last_update > 0.1:
            elapsed_time = time.time() - self.start_time
            eta = (elapsed_time / self.current_epoch) * (self.total_epochs - self.current_epoch)
            self.all_text = self.tmp_text + "Epochs: {}/{}, ETA: {} sec, Loss: {}, Accuracy: {}".format(
                self.current_epoch,
                self.total_epochs,
                round(eta, 1),
                round(self.logs.get('loss'), 3),
                round(self.logs.get('accuracy'), 3)
            )

            self.text_widget.delete('1.0', 'end')
            self.text_widget.insert('1.0', self.all_text)
            self.last_update = time.time()
    

    def on_train_end(self, logs=None):
        self.text_widget.insert('end', "\n")
        total_time = time.time() - self.start_time
        final_text = "Final Loss: {}, Final Accuracy: {}, Training time: {} sec".format(
            round(self.logs.get('loss'), 3),
            round(self.logs.get('accuracy'),3),
            round(total_time,1)
        )
        self.text_widget.insert('end', final_text)