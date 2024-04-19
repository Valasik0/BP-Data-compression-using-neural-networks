from tkinter import ttk
import tensorflow as tf
from tensorflow import keras
from TextLoader import *
from CustomModel import *
from TrainingProgress import *
from KthEntropyCalculator import *
from CompressedSize import *
from SequencesGenerator import *
import threading
import tkinter as tk
from tkinter import filedialog
from tkinter import PhotoImage
from TopFileFrame import *
import queue

class ModelFrame:
    def __init__(self, app):
        self.app = app	
        self.model_frame = None
        self.layer_type_combobox = None
        self.input_layer_var = None
        self.input_layer_set = False
        self.tree = None
        self.trash_icon = None
        self.model_label = None
        self.output_layer_item = None
        self.remove_button = None
        self.layer_settings_frame = None
        self.input_layer_checkbox = None
        self.layer_type_label = None
        self.neurons_label = None
        self.activation_label = None
        self.activation_combobox = None
        self.add_layer_button = None
        self.run_settings_frame = None
        self.batch_size_label = None
        self.batch_size_combobox = None
        self.batch_size_var = tk.StringVar()
        self.epochs_label = None
        self.epochs_scale = None
        self.run_button = None
        self.save_model_button = None
        self.batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        self.progress_window_training = None
        self.text_widget_trainig = None
        self.graph_widget = None

    def create_model_frame(self):
        self.model_frame = tk.Frame(self.app.root,
                                    borderwidth=1, 
                                    relief="solid",
                                    highlightbackground="#CCCCCC",
                                    highlightthickness=1, 
                                    bd=0)
        self.model_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew", rowspan=2)

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
        
        if "input" in self.tree.item(selected_item, "tags"):
            self.input_layer_set = False
            self.input_layer_var.set(False)
            self.input_layer_checkbox.config(state='normal')
            
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
                                            "Alphabet size (" + str(self.app.sigma_value.get()) + ")", 
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
        self.activation_combobox = ttk.Combobox(self.layer_settings_frame, values=["ReLU", "sigmoid", "tanh", "LeakyReLU"], state="readonly")
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

        self.save_model_button = ttk.Button(self.run_settings_frame, style="Custom2.TButton", text="Save model", command=lambda: self.save_model(self.app.global_model), state="disabled") 
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

    def build_model(self):
        if not self.input_layer_set:
            tk.messagebox.showerror("Error", "No input layer selected")
            return
        
        ta = TextAnalyzer(self.app.text_loader.loaded_text)

        if not ta.validate_text():
            return
        
        k = self.app.context_length_var.get()

        if not k:
            tk.messagebox.showinfo("Info", "No context lenght selected")
            return
        
        self.progress_window_training = tk.Toplevel(self.app.root)
        self.progress_window_training.title("Training progress")

        self.text_widget_trainig = tk.Text(self.progress_window_training)
        self.text_widget_trainig.grid(row=0, column=0)

        self.graph_widget = tk.Canvas(self.progress_window_training)
        

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

        self.sequence_generator = SequencesGenerator(self.app.text_loader.loaded_text, k, mapped_chars, sigma, batch_size)
        train_data_generator = self.sequence_generator.generate_sequences()

        progress_callback = TrainingProgress(self.text_widget_trainig, self.graph_widget, total_epochs)

        custom_model.fit(train_data_generator, 
                        epochs=total_epochs, 
                        steps_per_epoch=(len(self.app.text_loader.loaded_text)-k)//batch_size,
                        callbacks=[progress_callback]) 
        
        self.graph_widget.grid(row=0, column=1)

        self.global_model = custom_model 
        self.save_model_button.config(state="normal")