from tkinter import ttk
import tensorflow as tf
from tensorflow import keras
from TextLoader import *
from CustomModel import *
from TrainingProgress import *
from KthEntropyCalculator import *
from CompressedSize import *
import numpy as np
import threading
import tkinter as tk
from tkinter import filedialog
from collections import defaultdict

class MyApp:
    def __init__(self):
        self.text_loader = TextLoader()
        self.global_model = None
    
    def init_vars(self):
        self.context_length_var = tk.StringVar()
        self.compressed_size_var = tk.StringVar()
        self.kth_entropy_var = tk.StringVar()
        self.batch_compress_size_var = tk.StringVar()
        self.kth_entropy_var.set("Entropy: ")
        self.batch_size_var = tk.StringVar()
        self.text_size_var = tk.StringVar()
        self.text_size_var.set("Text size: ")
        self.sigma_var = tk.StringVar()
        self.sigma_var.set("Alphabet size: ")
        self.context_lengths = [1, 2, 3, 4, 5, 6, 7, 10, 15, 20]
        self.batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        self.batch_compress_sizes = [32, 64, 128, 256, 512, 1024, 2048]
        self.nodes_values = [32, 64, 128]

    

    def create_top_file_frame(self):
        self.top_file_frame = tk.Frame(self.root,
                                    borderwidth=1, 
                                    relief="solid",
                                    highlightbackground="#CCCCCC",
                                    highlightthickness=1, 
                                    bd=0)
        self.top_file_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew", columnspan=2)

    def create_model_frame(self):
        self.model_frame = tk.Frame(self.root,
                                    borderwidth=1, 
                                    relief="solid",
                                    highlightbackground="#CCCCCC",
                                    highlightthickness=1, 
                                    bd=0)
        self.model_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew", rowspan=2)

    def create_left_frame(self):
        self.left_frame = tk.Frame(self.root,
                                    borderwidth=1, 
                                    relief="solid",
                                    highlightbackground="#CCCCCC",
                                    highlightthickness=1, 
                                    bd=0)
        self.left_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

    def create_bottom_frame(self):
        self.bottom_frame = tk.Frame(self.root,
                        borderwidth=1, 
                        relief="solid",
                        highlightbackground="#CCCCCC",
                        highlightthickness=1, 
                                bd=0)
        self.bottom_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        self.bottom_frame.grid_columnconfigure(0, weight=1)

        self.entropy_title_label = tk.Label(self.bottom_frame, 
                            text="Text information",
                            font=("Arial", 12, "bold"))

        self.entropy_title_label.grid(row=0, column=0, pady=(5,15), sticky="ew")

        self.sigma_label = tk.Label(self.bottom_frame, textvariable=self.sigma_var)
        self.sigma_label.grid(row=1, column=0, padx=12, pady=7, sticky="w")

        self.text_size_label = tk.Label(self.bottom_frame, textvariable=self.text_size_var)
        self.text_size_label.grid(row=2, column=0, padx=12, pady=7, sticky="w")

        self.entropy_label = tk.Label(self.bottom_frame, textvariable=self.kth_entropy_var)
        self.entropy_label.grid(row=3, column=0, padx=12, pady=7, sticky="w")

        self.entropy_button = ttk.Button(self.bottom_frame, 
                                    style="Custom2.TButton",
                                    text="Calculate Entropy", 
                                    command=lambda: threading.Thread
                                    (target=self.kth_order_entropy, 
                                    args=(self.text_loader.loaded_text, self.context_length_var.get())).start())
        self.entropy_button.grid(row=4, column=0, pady=(5, 15), padx=22, sticky="w")


    def configure_frames(self):
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

    def create_top_file_frame_widgets(self):
        self.file_label = tk.Label(self.top_file_frame, text="File: ")
        self.file_label.grid(row=0, column=0, padx=(20, 20), sticky="w", pady=4)

        self.frame = tk.Frame(self.top_file_frame, 
                        borderwidth=1, 
                        relief="solid",
                        highlightbackground="#CCCCCC",
                        highlightthickness=1, 
                        bd=0,
                        )

        self.file_path_label = tk.Label(self.frame, 
                                text=self.text_loader.file_name, 
                                width=35,
                                anchor="w",
                                pady=4)

        self.frame.grid(row=0, column=1, padx=20, sticky="w")
        self.file_path_label.pack(side=tk.LEFT)

        self.load_text_button = ttk.Button(self.top_file_frame, 
                                    style="Custom.TButton",
                                    text="...", 
                                    command=lambda: (self.text_loader.load_file(), 
                                                    self.update_file_path_label()),
                                    width=5) 
        self.load_text_button.grid(row=0, column=2, padx=5, pady=4, sticky="w")


        self.context_length_label = tk.Label(self.top_file_frame, text="Context length:")
        self.context_length_label.grid(row=1, column=0, padx=20) 

        self.context_length_combobox = ttk.Combobox(self.top_file_frame, 
                                            textvariable=self.context_length_var, 
                                            values=self.context_lengths, 
                                            state="readonly",
                                            width=10)
        self.context_length_combobox.grid(row=1, column=1, pady=5, padx=(20, 0), sticky="w")

    def load_file(self):
        self.text_loader.load_file()
        self.update_file_path_label()
    
    def update_file_path_label(self):
        ta = TextAnalyzer(self.text_loader.loaded_text)
        self.text_size_var.set(f"Text size: {self.text_loader.file_size}")
        self.sigma_var.set(f"Alphabet size: {ta.sigma}")
        self.file_path_label.config(text=self.text_loader.file_name)
    
    def update_lstm_scales(self, value):
        for i, cb in enumerate(self.lstm_comboboxes):
            if i < int(value):
                cb.configure(state="readonly")
            else:
                cb.configure(state="disabled")

    def update_dense_scales(self, value):
        for i, cb in enumerate(self.dense_comboboxes):
            if i < int(value):
                cb.configure(state="readonly")
            else:
                cb.configure(state="disabled")

    def create_bottom_frame_widgets(self):
        self.bottom_frame.grid_columnconfigure(0, weight=1)

        self.entropy_title_label = tk.Label(self.bottom_frame, 
                            text="Text information",
                            font=("Arial", 12, "bold"))

        self.entropy_title_label.grid(row=0, column=0, pady=(5,15), sticky="ew")

        self.sigma_label = tk.Label(self.bottom_frame, textvariable=self.sigma_var)
        self.sigma_label.grid(row=1, column=0, padx=12, pady=7, sticky="w")

        self.text_size_label = tk.Label(self.bottom_frame, textvariable=self.text_size_var)
        self.text_size_label.grid(row=2, column=0, padx=12, pady=7, sticky="w")

        self.entropy_label = tk.Label(self.bottom_frame, textvariable=self.kth_entropy_var)
        self.entropy_label.grid(row=3, column=0, padx=12, pady=7, sticky="w")

        self.entropy_button = ttk.Button(self.bottom_frame, 
                                    style="Custom2.TButton",
                                    text="Calculate Entropy", 
                                    command=lambda: threading.Thread
                                    (target=self.kth_order_entropy, 
                                    args=(self.text_loader.loaded_text, self.context_length_var.get())).start())
        self.entropy_button.grid(row=4, column=0, pady=(5, 15), padx=22, sticky="w")

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
                                            args=(self.text_loader.loaded_text,
                                                self.context_length_var.get(),
                                                self.global_model)).start())
        compressed_size_button.grid(row=4, column=0, pady=(20, 10))

        compressed_size_label = tk.Label(self.left_frame, textvariable=self.compressed_size_var)
        compressed_size_label.grid(row=5, column=0, pady=5)


    def create_model_frame_widgets(self):
        self.model_label = tk.Label(self.model_frame, 
                       text="Model settings",
                       font=("Arial", 12, "bold"))

        self.model_label.grid(row=0, column=0, pady=5)

        self.lstm_frame = tk.Frame(self.model_frame,
                            borderwidth=1, 
                            relief="solid",
                            highlightbackground="#CCCCCC",
                            highlightthickness=1, 
                            bd=0)
        self.lstm_frame.grid(row=1, column=0, padx=(10,10), pady=(10, 0))

        self.dense_frame = tk.Frame(self.model_frame,
                            borderwidth=1, 
                            relief="solid",
                            highlightbackground="#CCCCCC",
                            highlightthickness=1, 
                            bd=0)
        self.dense_frame.grid(row=2, column=0, padx=(10,10), pady=(15, 20))

        self.lstm_label = tk.Label(self.lstm_frame, text="LSTM layers:")
        self.lstm_label.grid(row=0, column=0, columnspan=2)

        self.lstm_number_label = tk.Label(self.lstm_frame, text="Number")
        self.lstm_number_label.grid(row=1, column=0)

        self.lstm_nodes_label = tk.Label(self.lstm_frame, text="Nodes")
        self.lstm_nodes_label.grid(row=1, column=1)

        self.lstm_comboboxes = [ttk.Combobox(self.lstm_frame, values=self.nodes_values) for _ in range(2)]
        for i, combobox in enumerate(self.lstm_comboboxes):
            combobox.set(64)
            combobox.grid(row=i+3, column=1, pady=6, padx=7)
            if i != 0:  # Pouze první combobox bude mít readonly stav na začátku
                combobox.state(["disabled"])

        self.lstm_layers_scale = tk.Scale(self.lstm_frame, orient=tk.HORIZONTAL, from_=0, to=2, command=self.update_lstm_scales)
        self.lstm_layers_scale.grid(row=2, column=0, rowspan=2, padx=5)
        self.lstm_layers_scale.set(1)



        self.dense_label = tk.Label(self.dense_frame, text="Dense layers")
        self.dense_label.grid(row=0, column=0, columnspan=2)

        self.dense_number_label = tk.Label(self.dense_frame, text="Number")
        self.dense_number_label.grid(row=1, column=0)

        self.dense_nodes_label = tk.Label(self.dense_frame, text="Nodes")
        self.dense_nodes_label.grid(row=1, column=1)

        self.dense_comboboxes = [ttk.Combobox(self.dense_frame, values=self.nodes_values) for _ in range(2)]
        for i, combobox in enumerate(self.dense_comboboxes):
            combobox.set(64)
            combobox.grid(row=i+3, column=1, pady=6, padx=7)
            if i != 0:  # Pouze první combobox bude mít readonly stav na začátku
                combobox.state(["disabled"])

        self.dense_layers_scale = tk.Scale(self.dense_frame, 
                                    orient=tk.HORIZONTAL, 
                                    from_=0, 
                                    to=2, 
                                    command=self.update_dense_scales)

        self.dense_layers_scale.grid(row=2, column=0, rowspan=3, padx=5)
        self.dense_layers_scale.set(1)

        self.lstm_layers_scale.config(command=self.update_lstm_scales)
        self.dense_layers_scale.config(command=self.update_dense_scales)



        self.batch_size_label = tk.Label(self.model_frame, text="Batch size")
        self.batch_size_label.grid(row=3, column=0)

        self.batch_size_combobox = ttk.Combobox(self.model_frame, 
                                        values=self.batch_sizes, 
                                        state="readonly",
                                        textvariable=self.batch_size_var)
        self.batch_size_combobox.set(256)
        self.batch_size_combobox.grid(row=4, column=0, pady=2)


        self.epochs_label = tk.Label(self.model_frame, text="Total epochs")
        self.epochs_label.grid(row=5, column=0, pady=(20,0))

        self.epochs_scale = tk.Scale(self.model_frame, 
                                orient=tk.HORIZONTAL, 
                                from_=1, 
                                to=100,
                                length=200)
                                

        self.epochs_scale.set(10)
        self.epochs_scale.grid(row=6, column=0)

        self.run_button = ttk.Button(self.model_frame,
                                style="Custom2.TButton",
                                text="Run",
                                command=lambda: threading.Thread(target=self.build_model).start())

        self.run_button.grid(row=7, column=0, pady=15)

        self.save_model_button = ttk.Button(self.model_frame, 
                                    style="Custom2.TButton",
                                    text="Save model", 
                                    command=lambda: self.save_model(self.global_model),
                                    state="disabled") 
        self.save_model_button.grid(row=8, column=0, pady=5)
    
    def generate_sequences(self, text, k, mapped_chars, sigma, batch_size):
        while True:
            batch_chars = []
            batch_labels = []

            for i in range(k, len(text)):
                sequence = text[i - k:i + 1] # k = 3, text = abrakadabra, abra, brak, raka, akad .....
                mapped_sequence = [mapped_chars[char] for char in sequence] # [1,2,3,1], [2,3,1,4], [3,1,4,1], [1,4,1,5] ... namapuje znaky na cisla
                train_chars = tf.keras.utils.to_categorical(mapped_sequence[:-1], num_classes=sigma)
                train_label = tf.keras.utils.to_categorical(mapped_sequence[-1], num_classes=sigma)
                batch_chars.append(train_chars)
                batch_labels.append(train_label)

                if len(batch_chars) == batch_size:
                    yield np.array(batch_chars), np.array(batch_labels)
                    batch_chars = []
                    batch_labels = []

            if batch_chars:  # yield zbývajících znaků na konci batche
                yield np.array(batch_chars), np.array(batch_labels)


    def save_model(self, model):
        file_path = filedialog.asksaveasfilename()
        if file_path:
            try:
                model.save(file_path)
            except Exception as e:
                print(f"An error occurred: {e}")

    def load_model(self):
        directory_path = filedialog.askdirectory()
        if directory_path:
            try:
                self.global_model = keras.models.load_model(directory_path)
            except (OSError, ValueError) as e:
                tk.messagebox.showerror("Error", f"Error loading model: {e}")
            return None
    
    def kth_order_entropy(self, text, k):
        ta = TextAnalyzer(text)
        if not ta.validate_text():
            return
        
        if not k:
            tk.messagebox.showinfo("Info", "No context lenght selected")
            return

        self.kth_entropy_var.set("Entropy: calculating...")
        kth_entropy_calculator = KthEntropyCalculator(text, int(k))
        entropy = kth_entropy_calculator.calculate_kth_entropy()

        self.entropy_label.config(text=str(round(entropy, 3)))

        self.kth_entropy_var.set(f"Entropy: {round(entropy, 3)} bpB")

    def estimated_compressed_size(self, text, k, model):
        if not k:
            tk.messagebox.showinfo("Info", "No context lenght selected")
            return
        
        self.progress_window_compress = tk.Toplevel(self.root)
        self.progress_window_compress.title("Computing compressed size")
        self.progress_window_compress.geometry("400x200")

        self.text_widget_compress = tk.Text(self.progress_window_compress)
        self.text_widget_compress.pack()


        batch_size = int(self.batch_compress_size_var.get())
        compessed_size_calculator = CompressedSize(model, int(k), batch_size, text, self.text_widget_compress)
        self.compressed_size_var.set("calculating...")
        compressed_size = compessed_size_calculator.compute(text)
        
        

        if compressed_size is not None:
            compressed_size = compressed_size / (1024 * 1024 * 8)
            print(f"Estimated compressed size: {compressed_size} MB")
            self.compressed_size_var.set(f"Size: {round(compressed_size, 3)} MB")
        else:
            self.compressed_size_var.set("Size: error")
        self.progress_window_compress.destroy()

    def build_model(self):
        ta = TextAnalyzer(self.text_loader.loaded_text)

        if not ta.validate_text():
            return
        
        k = self.context_length_var.get()

        if not k:
            tk.messagebox.showinfo("Info", "No context lenght selected")
            return
        
        self.progress_window_training = tk.Toplevel(self.root)
        self.progress_window_training.title("Training progress")

        self.text_widget_trainig = tk.Text(self.progress_window_training)
        self.text_widget_trainig.pack()

    

        k = int(k)
        mapped_chars = ta.compute_mapped_chars()
        sigma = ta.compute_unique_chars()
        num_dense_layers = int(self.dense_layers_scale.get())
        dense_layer_sizes = [int(self.dense_comboboxes[i].get()) for i in range(num_dense_layers)]
        num_lstm_layers = int(self.lstm_layers_scale.get())
        lstm_layer_sizes = [int(self.lstm_comboboxes[i].get()) for i in range(num_lstm_layers)]

        custom_model = CustomModel(io_size=sigma, 
                                k=k, 
                                num_dense_layers=num_dense_layers, 
                                dense_layer_sizes=dense_layer_sizes, 
                                num_lstm_layers=num_lstm_layers, 
                                lstm_layer_sizes=lstm_layer_sizes)

        custom_model.compile(optimizer="adam", 
                            loss="categorical_crossentropy", 
                            metrics=['accuracy']) 

        batch_size = int(self.batch_size_var.get())
        total_epochs = self.epochs_scale.get()

        train_data_generator = self.generate_sequences(self.text_loader.loaded_text, k, mapped_chars, sigma, batch_size)

        progress_callback = TrainingProgress(self.text_widget_trainig, total_epochs)

        custom_model.fit(train_data_generator, 
                        epochs=total_epochs, 
                        steps_per_epoch=(len(self.text_loader.loaded_text)-k)//batch_size,
                        callbacks=[progress_callback]) 

        self.global_model = custom_model 
        self.save_model_button.config(state="normal")

    def run(self):
        self.root = tk.Tk()
        self.root.title("BP")

        self.init_vars()
        self.create_top_file_frame()
        self.create_model_frame()
        self.create_left_frame()
        self.create_bottom_frame()
        self.configure_frames()
        self.create_top_file_frame_widgets()
        self.create_model_frame_widgets()
        self.create_left_frame_widgets()
        self.create_bottom_frame_widgets()

        self.root.mainloop()