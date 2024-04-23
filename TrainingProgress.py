import tensorflow as tf
import tkinter as tk
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class TrainingProgress(tf.keras.callbacks.Callback):
    def __init__(self, text_widget,graph_widget, total_epochs):
        super().__init__()
        self.text_widget = text_widget
        self.graph_widget = graph_widget
        self.total_epochs = total_epochs
        self.start_time = time.time()
        self.last_update = time.time()
        self.logs = {}
        self.current_epoch = 0
        self.all_text = ""
        self.tmp_text = ""
        self.loss_history = [] 
        

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch + 1
    
    def on_epoch_end(self, epoch, logs=None):
        self.all_text = self.all_text + '\n'
        self.tmp_text = self.all_text
        self.loss_history.append(logs['loss'])
        
    
    def on_train_batch_end(self, batch, logs=None):
        self.logs = logs or {}
        if time.time() - self.last_update > 0.1:
            elapsed_time = time.time() - self.start_time
            eta = (elapsed_time / self.current_epoch) * (self.total_epochs - self.current_epoch)
            self.all_text = self.tmp_text + "Epochs: {}/{}, ETA: {}, Loss: {}, Accuracy: {}".format(
                self.current_epoch,
                self.total_epochs,
                self.format_time(round(eta, 1)),
                round(self.logs.get('loss'), 3),
                round(self.logs.get('accuracy'), 3)
            )
            if self.text_widget.winfo_exists():
                self.text_widget.delete('1.0', 'end')
                self.text_widget.insert('1.0', self.all_text)
            self.last_update = time.time()
    

    def on_train_end(self, logs=None):
        if self.text_widget.winfo_exists():
            self.text_widget.insert('end', "\n")
            total_time = time.time() - self.start_time
            final_text = "Completed. Final Loss: {}, Final Accuracy: {}, Training time: {}".format(
                round(self.logs.get('loss'), 3),
                round(self.logs.get('accuracy'),3),
                self.format_time(round(total_time,1))
            )
            self.text_widget.insert('end', final_text)

        if self.graph_widget.winfo_exists():
            self.graph_widget.after(100, self.draw_graph)

    def draw_graph(self):
        if self.text_widget.winfo_exists() and self.graph_widget.winfo_exists():
            widget_height = self.text_widget.winfo_height()
            fig_height = widget_height / self.graph_widget.winfo_fpixels('1i')

            self.fig, self.ax = plt.subplots(figsize=(fig_height, fig_height))
            self.line, = self.ax.plot(self.loss_history)

            self.ax.set_xlabel('Epochs')
            self.ax.set_ylabel('Loss')
            self.ax.set_title('Training Loss')

            self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_widget)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack()

    def format_time(self, seconds):
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return "{}h {}m {}s".format(int(hours), int(minutes), int(seconds))
    