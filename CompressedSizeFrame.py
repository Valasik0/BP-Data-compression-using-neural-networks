import math
from tkinter import ttk
from tensorflow import keras
from TextLoader import *
from CustomModel import *
from TrainingProgress import *
from CompressedSize import *
from SequencesGenerator import *
import threading
import tkinter as tk
from tkinter import filedialog
from TopFileFrame import *

class CompressedSizeFrame:
    def __init__(self, app):
        self.app = app
        self.left_frame = None
        self.compressed_size_title_label = None
        self.load_model_button = None
        self.batch_compress_size_label = None
        self.batch_compress_size_combobox = None
        self.compressed_size_var = tk.StringVar()
        self.batch_compress_sizes = [32, 64, 128, 256, 512, 1024, 2048]
        self.batch_compress_size_var = tk.StringVar()
        self.compressed_size_button = None
        self.compressed_size_label = None
        self.progress_window_compress = None
        self.text_widget_compress = None

    def create_left_frame(self):
        self.left_frame = tk.Frame(self.app.root,
                                    borderwidth=1, 
                                    relief="solid",
                                    highlightbackground="#CCCCCC",
                                    highlightthickness=1, 
                                    bd=0)
        self.left_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

    def create_left_frame_widgets(self):
        self.compressed_size_title_label = tk.Label(self.left_frame, 
                            text="Compressed size",
                            font=("Arial", 12, "bold"))

        self.compressed_size_title_label.grid(row=0, column=0, pady=(5,15))

        self.load_model_button = ttk.Button(self.left_frame, 
                                    style="Custom2.TButton",
                                    text="Load model", 
                                    command=lambda: self.load_model())
        self.load_model_button.grid(row=1, column=0, pady=(5,15))

        self.batch_compress_size_label = tk.Label(self.left_frame, text="Batch size")
        self.batch_compress_size_label.grid(row=2, column=0, pady=(3,0))


        self.batch_compress_size_combobox = ttk.Combobox(self.left_frame, 
                                        values=self.batch_compress_sizes, 
                                        state="readonly",
                                        textvariable=self.batch_compress_size_var)
        self.batch_compress_size_combobox.set(256)
        self.batch_compress_size_combobox.grid(row=3, column=0, pady=(0,5), padx=20)

        compressed_size_button = ttk.Button(self.left_frame,
                                            style="Custom2.TButton",
                                            text="Compute",
                                            command=lambda: threading.Thread
                                            (target=self.estimated_compressed_size,
                                            args=(self.app.text_loader.loaded_text,
                                                self.app.context_length_var.get(),
                                                self.app.global_model)).start())
        compressed_size_button.grid(row=4, column=0, pady=(20, 10))

        compressed_size_label = tk.Label(self.left_frame, textvariable=self.compressed_size_var)
        compressed_size_label.grid(row=5, column=0, pady=5)

    def load_model(self):
        directory_path = filedialog.askdirectory()
        if directory_path:
            try:
                self.app.global_model = keras.models.load_model(directory_path)

                if isinstance(self.app.global_model, keras.models.Sequential):
                    layers = self.app.global_model.layers
                else:
                    layers = self.app.global_model.layers[0].layers

                total_layers = len(layers)

                for i in self.app.model_frame.tree.get_children():
                    self.app.model_frame.tree.delete(i)

                # Add the layers to the tree
                for index, layer in enumerate(layers):
                    layer_type = layer.__class__.__name__

                    if layer_type == 'Flatten':
                        activation = "None"
                        neurons = "None"
                    else:
                        layer_config = layer.get_config()
                        activation = layer_config.get("activation", "None")
                        neurons = layer_config.get("units", "None")

                    if index == 0:
                        layer_layout = "Input Layer"
                        tag = "input"
                        self.app.context_length_var.set(str(layer.input_shape[1]))
                        print(layer.input_shape[1])
                    elif index == total_layers - 1:
                        layer_layout = "Output Layer"
                        tag = "output"
                    else:
                        layer_layout = "Hidden Layer"
                        tag = "hidden"

                    
                    self.app.model_frame.tree.insert('', 'end', values=(layer_layout, layer_type, neurons, activation), tags=(tag,))
                    self.app.model_frame.input_layer_set = True

            except (OSError, ValueError) as e:
                tk.messagebox.showerror("Error", f"Error loading model: {e}")
            return None
        
    def estimated_compressed_size(self, text, k, model):
        if not k:
            tk.messagebox.showinfo("Info", "No context lenght selected")
            return
        
        self.progress_window_compress = tk.Toplevel(self.app.root)
        self.progress_window_compress.title("Computing compressed size")
        self.progress_window_compress.geometry("400x200")
        self.progress_window_compress.resizable(False, False)

        self.text_widget_compress = tk.Text(self.progress_window_compress,  width=50, height=9)
        self.text_widget_compress.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        


        batch_size = int(self.batch_compress_size_var.get())
        compressed_size_calculator = CompressedSize(model, int(k), batch_size, text, self.text_widget_compress)
        
        cancel_button = tk.Button(self.progress_window_compress, text="Cancel",command= lambda:compressed_size_calculator.cancel_computation(self.progress_window_compress))
        cancel_button.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        self.compressed_size_var.set("calculating...")
        compressed_size = compressed_size_calculator.compute(text)
        
        

        if compressed_size is not None:
            compressed_size = compressed_size / 8
            print(f"Estimated compressed size: {self.convert_size(compressed_size)}")
            self.compressed_size_var.set(f"Size: {self.convert_size(compressed_size)}")
        else:
            self.compressed_size_var.set("Size: error")
            self.progress_window_compress.destroy()


    def convert_size(self, size_bytes):
        if size_bytes == 0:
            return "0B"
        size_name = ("B", "kB", "MB", "GB")
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_name[i]}"