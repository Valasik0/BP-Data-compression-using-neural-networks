from tkinter import ttk
import tensorflow as tf
from tensorflow import keras
from TextLoader import *
from CustomModel import *
from TrainingProgress import *
from KthEntropyCalculator import *
from CompressedSize import *
from SequencesGenerator import *
import numpy as np
import threading
import tkinter as tk
from tkinter import filedialog
from tkinter import PhotoImage

class MyApp:
    def __init__(self):
        self.text_loader = TextLoader()
        self.global_model = None
        self.sequence_generator = None
        self.input_layer_set = False
    
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
        self.sigma_value = tk.IntVar()
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
                                width=45,
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
        self.sigma_value.set(ta.sigma)
        self.text_size_var.set(f"Text size: {self.text_loader.file_size}")
        self.sigma_var.set(f"Alphabet size: {self.sigma_value.get()}")
        self.file_path_label.config(text=self.text_loader.file_name)
        self.tree.set(self.output_layer_item, column=2, 
                      value="Alphabet size (" + str(self.sigma_value.get()) + ")")

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

    def validate_neuron_count(self, new_text):
        if not new_text or new_text == "None":  # the field is being cleared
            return True

        try:
            value = int(new_text)
            if value < 1 or value > 2048:
                messagebox.showerror("Invalid input", "Number of units must be an integer between 1 and 2048.")
                return False
        except ValueError:
            messagebox.showerror("Invalid input", "Number of units must be an integer between 1 and 2048.")
            return False

        return True
    
    def add_layer(self):
        layer_layout = "Hidden Layer"
        neurons = self.neurons_entry.get()
        layer_type = self.layer_type_combobox.get()

        if layer_type != 'Flatten' and not neurons:
            messagebox.showerror("Invalid input", "Number of neurons cannot be empty.")
            return

        if self.input_layer_var.get() and self.input_layer_set:
            messagebox.showerror("Invalid input", "Input layer has already been set.")
            return
        elif self.input_layer_var.get():
            layer_layout = "Input Layer"
            self.input_layer_set = True
            self.input_layer_var.set(False)
            self.input_layer_checkbox.config(state='disabled')

        
        layer_type = self.layer_type_combobox.get()
        
        activation = self.activation_combobox.get()

        

        if layer_layout == "Input Layer":
            self.tree.insert('', 0, values=(layer_layout, layer_type, neurons, activation), tags=("input",))
        else:
            output_layer = self.tree.get_children()[-1]
            self.tree.insert('', self.tree.index(output_layer), values=(layer_layout, layer_type, neurons, activation), tags=("hidden",))

    def remove_layer(self):
        selected_item = self.tree.selection()
        if "output" in self.tree.item(selected_item, "tags"):
            messagebox.showerror("Invalid operation", "Cannot remove the output layer.")
            return
        self.tree.delete(selected_item)   

    def on_layer_type_changed(self, event):
        # Get the current layer type
        layer_type = self.layer_type_combobox.get()

        # If the layer type is 'Flatten', disable the neurons entry and activation combobox
        if layer_type == 'Flatten':
            self.neurons_entry.delete(0, 'end')
            self.neurons_entry.insert(0, 'None')
            self.neurons_entry.config(state='disabled')
            self.activation_combobox.config(state='disabled')
            self.input_layer_checkbox.config(state='disabled')
            self.activation_combobox.set('None')
            
        else:
            self.neurons_entry.config(state='normal')
            self.activation_combobox.config(state='readonly')
            self.activation_combobox.set('relu')
            self.neurons_entry.delete(0, 'end')

    def create_model_frame_widgets(self):
        self.trash_icon = PhotoImage(file="img\860829.png")

        self.model_label = tk.Label(self.model_frame, 
                    text="Model settings",
                    font=("Arial", 12, "bold"))
        self.model_label.grid(row=0, column=0, pady=5, columnspan=3)

        # Create the Treeview widget
        self.tree = ttk.Treeview(self.model_frame)
        self.tree["columns"] = ("layer", "type", "units", "activation")
        self.tree.column("#0", width=0)
        self.tree.column("layer", width=140, anchor='center')
        self.tree.column("type", width=140, anchor='center')
        self.tree.column("units", width=140, anchor='center')
        self.tree.column("activation", width=140, anchor='center')
        self.tree.heading("layer", text="Layer")
        self.tree.heading("type", text="Type")
        self.tree.heading("units", text="Units")
        self.tree.heading("activation", text="Activation")
        self.tree.grid(row=1, column=0, pady=5, padx=(5, 0), columnspan=2)  # Adjust the row and column as needed
        self.output_layer_item = self.tree.insert('', 'end', values=("Output Layer", 
                                            "Dense", 
                                            "Alphabet size (" + str(self.sigma_value.get()) + ")", 
                                            "softmax"), 
                                            tags=("output",))

        # Change the background color and font color
        style = ttk.Style()
        style.configure("Treeview",
                        background="lightgrey",
                        foreground="black",
                        rowheight=25,
                        fieldbackground="lightgrey")
        style.map('Treeview', background=[('selected', '#828282')])


        # Nastavení barev podle značek
        self.tree.tag_configure("input", background="#FFA3A3")
        self.tree.tag_configure("hidden", background="lightgrey")
        self.tree.tag_configure("output", background="#8AB9FF")

        self.remove_button = tk.Button(self.model_frame, image=self.trash_icon, command=self.remove_layer)
        
        self.remove_button.grid(row=1, column=2, padx=5, pady=5, sticky='sw')

        # Create the layer settings frame
        self.layer_settings_frame = tk.Frame(self.model_frame,
                                             borderwidth=1, 
                                             relief="solid",
                                             highlightbackground="#CCCCCC",
                                             highlightthickness=1, 
                                             bd=0)
        self.layer_settings_frame.grid(row=2, column=0, pady=5, padx=5, sticky="nsew")

        self.input_layer_var = tk.IntVar()
        self.input_layer_checkbox = tk.Checkbutton(self.layer_settings_frame, text="Set as input layer", variable=self.input_layer_var)
        self.input_layer_checkbox.grid(row=4, column=0, columnspan=2, sticky="w", padx=10, pady=8)

        self.layer_type_label = tk.Label(self.layer_settings_frame, text="Layer type:")
        self.layer_type_label.grid(row=0, column=0, sticky="w", padx = 10, pady=8)

        self.layer_type_combobox = ttk.Combobox(self.layer_settings_frame, values=["LSTM", "Dense", "Flatten"], state="readonly")
        self.layer_type_combobox.set("LSTM")
        self.layer_type_combobox.grid(row=0, column=1, sticky="w", pady=8, padx=(0, 10))
        self.layer_type_combobox.bind('<<ComboboxSelected>>', self.on_layer_type_changed)

        self.neurons_label = tk.Label(self.layer_settings_frame, text="Number of units:")
        self.neurons_label.grid(row=1, column=0, sticky="w", padx = 10, pady=8)

        validation = self.layer_settings_frame.register(self.validate_neuron_count)
        self.neurons_entry = tk.Entry(self.layer_settings_frame, validate="key", validatecommand=(validation, '%P'))
        self.neurons_entry.grid(row=1, column=1, sticky="w", pady=8)

        self.activation_label = tk.Label(self.layer_settings_frame, text="Activation function:")
        self.activation_label.grid(row=2, column=0, sticky="w", padx = 10, pady=5)
        self.activation_combobox = ttk.Combobox(self.layer_settings_frame, values=["relu", "sigmoid", "tanh"], state="readonly")
        self.activation_combobox.set("relu")
        self.activation_combobox.grid(row=2, column=1, sticky="w", pady=8, padx=(0, 10))

        self.add_layer_button = ttk.Button(self.layer_settings_frame, text="Add Layer", command=self.add_layer)
        self.add_layer_button.grid(row=3, column=0, columnspan=2, sticky="w", padx=10, pady=8)

        # Create the run settings frame
        self.run_settings_frame = tk.Frame(self.model_frame,
                                           borderwidth=1, 
                                            relief="solid",
                                            highlightbackground="#CCCCCC",
                                            highlightthickness=1, 
                                            bd=0)
        self.run_settings_frame.grid(row=2, column=1, columnspan=2, pady=5, padx=5, sticky="nswe")

        self.batch_size_label = tk.Label(self.run_settings_frame, text="Batch size:")
        self.batch_size_label.grid(row=0, column=0, sticky="w", pady=8, padx=12)

        self.batch_size_combobox = ttk.Combobox(self.run_settings_frame, values=self.batch_sizes, state="readonly", textvariable=self.batch_size_var)
        self.batch_size_combobox.set(256)
        self.batch_size_combobox.grid(row=0, column=1, sticky="w", pady=8)

        self.epochs_label = tk.Label(self.run_settings_frame, text="Total epochs:")
        self.epochs_label.grid(row=1, column=0, sticky="w", pady=(15, 0), padx=10)

        self.epochs_scale = tk.Scale(self.run_settings_frame, orient=tk.HORIZONTAL, from_=1, to=100, length=140)
        self.epochs_scale.set(10)
        self.epochs_scale.grid(row=1, column=1, sticky="w", pady=8)

        self.run_button = ttk.Button(self.run_settings_frame, style="Custom2.TButton", text="Run", command=lambda: threading.Thread(target=self.build_model).start())
        self.run_button.grid(row=2, column=0, columnspan=2, sticky="w", pady=8, padx=10)

        self.save_model_button = ttk.Button(self.run_settings_frame, style="Custom2.TButton", text="Save model", command=lambda: self.save_model(self.global_model), state="disabled") 
        self.save_model_button.grid(row=3, column=0, columnspan=2, sticky="w", pady=8, padx=10)

        self.model_frame.columnconfigure(0, weight=1)
        self.model_frame.columnconfigure(1, weight=1)
        self.model_frame.rowconfigure(2, weight=1)

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
                total_layers = len(self.global_model.layers)

                for i in self.tree.get_children():
                    self.tree.delete(i)

                # Add the layers to the tree
                for index, layer in enumerate(self.global_model.layers):
                    layer_type = layer.__class__.__name__

                    if layer_type == 'Flatten':
                        activation = "None"
                        neurons = "None"
                    else:
                        activation = layer.get_config()["activation"]
                        neurons = layer.get_config()["units"]

                    if index == 0:
                        layer_layout = "Input Layer"
                        tag = "input"
                        self.context_length_var.set(str(layer.input_shape[1]))
                        print(layer.input_shape[1])
                    elif index == total_layers - 1:
                        layer_layout = "Output Layer"
                        tag = "output"
                    else:
                        layer_layout = "Hidden Layer"
                        tag = "hidden"

                    
                    self.tree.insert('', 'end', values=(layer_layout, layer_type, neurons, activation), tags=(tag,))


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
        if not self.input_layer_set:
            tk.messagebox.showerror("Error", "No input layer selected")
            return
        
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

        # Get the layers from the table
        layers = []
        items = self.tree.get_children()[:-1]
        for item in items:
            _, layer_type, neurons, activation = self.tree.item(item, "values")
            if neurons == 'None':
                neurons = 0
            layers.append((layer_type, int(neurons), activation))   

        # Create the model with the layers
        custom_model = CustomModel(layers, k, sigma)

        custom_model.build((None, k, sigma))
        custom_model.summary()

        custom_model.compile(optimizer="adam", 
                            loss="categorical_crossentropy", 
                            metrics=['accuracy']) 
        
        

        batch_size = int(self.batch_size_var.get())
        total_epochs = self.epochs_scale.get()

        self.sequence_generator = SequencesGenerator(self.text_loader.loaded_text, k, mapped_chars, sigma, batch_size)
        train_data_generator = self.sequence_generator.generate_sequences()

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
        self.root.resizable(False, False)

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