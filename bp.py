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

#https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
#3.6771007487841416 MB k2
#3.0428894963115454 MB k3
#4.1269525192910805 MB k4
#2.6320016653917264 MB k5
#2.5886756268446334 MB k7
#2.578941140702227 MB k10
#2.6362323465000372 MB k15


def validate_text(text):
    if not isinstance(text, bytes):
        messagebox.showerror("Error", "Loaded text is in incorrect format.")
        return False
    if len(text) == 0:
        messagebox.showinfo("Info", "Loaded text is empty. Please load a text.")
        return False
    return True

def map_chars(text):
    if validate_text(text):
        chars = sorted(list(set(text))) #chars - unikatni znaky v textu pomoci metody set
        mapped_chars = {c:i for i, c in enumerate(chars)}

        mapped_chars['<UNK>'] = len(mapped_chars)
        return mapped_chars
    
    else:
        return None

def count_unique_chars(mapped_chars): 
    if mapped_chars is None or len(mapped_chars) == 0:
        return None
    
    return len(mapped_chars)

def generate_sequences(text, k, mapped_chars, sigma, batch_size):
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

def save_model(model):
    file_path = filedialog.asksaveasfilename()
    if file_path:
        try:
            model.save(file_path)
        except Exception as e:
            print(f"An error occurred: {e}")

def load_model():
    directory_path = filedialog.askdirectory()
    if directory_path:
        try:
            global global_model
            global_model = keras.models.load_model(directory_path)
        except (OSError, ValueError) as e:
            tk.messagebox.showerror("Error", f"Error loading model: {e}")
            return None

def kth_order_entropy(text, k):
    if not validate_text(text):
        return
    
    if not k:
        tk.messagebox.showinfo("Info", "No context lenght selected")
        return

    kth_entropy_var.set("Entropy: calculating...")
    kth_entropy_calculator = KthEntropyCalculator(text, int(k))
    entropy = kth_entropy_calculator.calculate_kth_entropy()

    entropy_label.config(text=str(round(entropy, 3)))

    kth_entropy_var.set(f"Entropy: {round(entropy, 3)} bpB")

def estimated_compressed_size(text, k, model):
    if not k:
        tk.messagebox.showinfo("Info", "No context lenght selected")
        return

    batch_size = int(batch_compress_size_var.get())
    compessed_size_calculator = CompressedSize(model, int(k), batch_size, text)
    compressed_size_var.set("calculating...")
    compressed_size = compessed_size_calculator.compute(text)
    

    if compressed_size is not None:
        compressed_size = compressed_size / (1024 * 1024 * 8)
        print(f"Estimated compressed size: {compressed_size} MB")
        compressed_size_var.set(f"Size: {round(compressed_size, 3)} MB")
    else:
        compressed_size_var.set("Size: error")

def build_model():
    if not validate_text(text_loader.loaded_text):
        return
    
    k = context_length_var.get()

    if not k:
        tk.messagebox.showinfo("Info", "No context lenght selected")
        return
    
    progress_window = tk.Toplevel(root)
    progress_window.title("Training progress")

    text_widget = tk.Text(progress_window)
    text_widget.pack()
    k = int(k)
    mapped_chars = map_chars(text_loader.loaded_text)
    sigma = count_unique_chars(mapped_chars)
    num_dense_layers = int(dense_layers_scale.get())
    dense_layer_sizes = [int(dense_comboboxes[i].get()) for i in range(num_dense_layers)]
    num_lstm_layers = int(lstm_layers_scale.get())
    lstm_layer_sizes = [int(lstm_comboboxes[i].get()) for i in range(num_lstm_layers)]

    custom_model = CustomModel(io_size=sigma, 
                               k=k, 
                               num_dense_layers=num_dense_layers, 
                               dense_layer_sizes=dense_layer_sizes, 
                               num_lstm_layers=num_lstm_layers, 
                               lstm_layer_sizes=lstm_layer_sizes)

    custom_model.compile(optimizer="adam", 
                          loss="categorical_crossentropy", 
                          metrics=['accuracy']) 

    batch_size = int(batch_size_var.get())
    total_epochs = epochs_scale.get()

    train_data_generator = generate_sequences(text_loader.loaded_text, k, mapped_chars, sigma, batch_size)

    progress_callback = TrainingProgress(text_widget, total_epochs)

    custom_model.fit(train_data_generator, 
                     epochs=total_epochs, 
                     steps_per_epoch=(len(text_loader.loaded_text)-k)//batch_size,
                     callbacks=[progress_callback]) 

    global global_model
    global_model = custom_model 
    save_model_button.config(state="normal")
    

def update_file_path_label():
    ta = TextAnalyzer(text_loader.loaded_text)
    text_size_var.set(f"Text size: {text_loader.file_size}")
    sigma_var.set(f"Alphabet size: {ta.sigma}")
    file_path_label.config(text=text_loader.file_name)
    
    

def update_lstm_scales(value):
    for i, cb in enumerate(lstm_comboboxes):
        if i < int(value):
            cb.configure(state="readonly")
        else:
            cb.configure(state="disabled")

def update_dense_scales(value):
    for i, cb in enumerate(dense_comboboxes):
        if i < int(value):
            cb.configure(state="readonly")
        else:
            cb.configure(state="disabled")
        
text_loader = TextLoader()
global_model = None

root = tk.Tk()
"""
screen_width = 1000 #root.winfo_screenwidth()
screen_height = 800 #root.winfo_screenheight()

root.geometry(f"{screen_width}x{screen_height}")
"""
root.title("BP")


button_load_style = ttk.Style()
button_load_style.configure("Custom.TButton", 
                foreground="black", 
                font=("Segoe UI", 12, "bold"), 
                borderwidth=2)


button_style = ttk.Style()
button_style.configure("Custom2.TButton", 
                foreground="black", 
                font=("Arial", 8,), 
                borderwidth=2,
                padding=(0, 4, 0, 4),
                width=20)


context_lengths = [2, 3, 4, 5, 6, 7, 10, 15, 20]
context_length_var = tk.StringVar()

batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
batch_size_var = tk.StringVar(root)

batch_compress_sizes = [32, 64, 128, 256, 512, 1024, 2048]
batch_compress_size_var = tk.StringVar(root)

nodes_values = [32, 64, 128]

compressed_size_var = tk.StringVar()

kth_entropy_var = tk.StringVar()
kth_entropy_var.set("Entropy: ")

text_size_var = tk.StringVar()
text_size_var.set("Text size: ")

sigma_var = tk.StringVar()
sigma_var.set("Alphabet size: ")

# Top file frame
top_file_frame = tk.Frame(root,
                          borderwidth=1, 
                          relief="solid",
                          highlightbackground="#CCCCCC",
                          highlightthickness=1, 
                          bd=0)
top_file_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew", columnspan=2)

# Model frame
model_frame = tk.Frame(root,
                       borderwidth=1, 
                       relief="solid",
                       highlightbackground="#CCCCCC",
                       highlightthickness=1, 
                       bd=0)
model_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew", rowspan=2)

# Configure the grid to expand the frames
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(2, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)


#-----------------top file frame-----------------
file_label = tk.Label(top_file_frame, text="File: ")
file_label.grid(row=0, column=0, padx=(20, 20), sticky="w", pady=4)

frame = tk.Frame(top_file_frame, 
                 borderwidth=1, 
                 relief="solid",
                 highlightbackground="#CCCCCC",
                 highlightthickness=1, 
                 bd=0,
                 )

file_path_label = tk.Label(frame, 
                           text=text_loader.file_name, 
                           width=35,
                           anchor="w",
                           pady=4)

frame.grid(row=0, column=1, padx=20, sticky="w")
file_path_label.pack(side=tk.LEFT)

load_text_button = ttk.Button(top_file_frame, 
                             style="Custom.TButton",
                             text="...", 
                             command=lambda: (text_loader.load_file(), 
                                              update_file_path_label()),
                             width=5) 
load_text_button.grid(row=0, column=2, padx=5, pady=4, sticky="w")


context_length_label = tk.Label(top_file_frame, text="Context length:")
context_length_label.grid(row=1, column=0, padx=20) 

context_length_combobox = ttk.Combobox(top_file_frame, 
                                       textvariable=context_length_var, 
                                       values=context_lengths, 
                                       state="readonly",
                                       width=10)
context_length_combobox.grid(row=1, column=1, pady=5, padx=(20, 0), sticky="w")

#------------------------------------------------


#-----------------model frame---------------------
model_label = tk.Label(model_frame, 
                       text="Model settings",
                       font=("Arial", 12, "bold"))

model_label.grid(row=0, column=0, pady=5)

lstm_frame = tk.Frame(model_frame,
                      borderwidth=1, 
                      relief="solid",
                      highlightbackground="#CCCCCC",
                      highlightthickness=1, 
                      bd=0)
lstm_frame.grid(row=1, column=0, padx=(10,10), pady=(10, 0))

dense_frame = tk.Frame(model_frame,
                       borderwidth=1, 
                       relief="solid",
                       highlightbackground="#CCCCCC",
                       highlightthickness=1, 
                       bd=0)
dense_frame.grid(row=2, column=0, padx=(10,10), pady=(15, 20))

lstm_label = tk.Label(lstm_frame, text="LSTM layers:")
lstm_label.grid(row=0, column=0, columnspan=2)

lstm_number_label = tk.Label(lstm_frame, text="Number")
lstm_number_label.grid(row=1, column=0)

lstm_nodes_label = tk.Label(lstm_frame, text="Nodes")
lstm_nodes_label.grid(row=1, column=1)

lstm_comboboxes = [ttk.Combobox(lstm_frame, values=nodes_values) for _ in range(2)]
for i, combobox in enumerate(lstm_comboboxes):
    combobox.set(64)
    combobox.grid(row=i+3, column=1, pady=6, padx=7)
    if i != 0:  # Pouze první combobox bude mít readonly stav na začátku
        combobox.state(["disabled"])

lstm_layers_scale = tk.Scale(lstm_frame, orient=tk.HORIZONTAL, from_=0, to=2, command=update_lstm_scales)
lstm_layers_scale.grid(row=2, column=0, rowspan=2, padx=5)
lstm_layers_scale.set(1)



dense_label = tk.Label(dense_frame, text="Dense layers")
dense_label.grid(row=0, column=0, columnspan=2)

dense_number_label = tk.Label(dense_frame, text="Number")
dense_number_label.grid(row=1, column=0)

dense_nodes_label = tk.Label(dense_frame, text="Nodes")
dense_nodes_label.grid(row=1, column=1)

dense_comboboxes = [ttk.Combobox(dense_frame, values=nodes_values) for _ in range(2)]
for i, combobox in enumerate(dense_comboboxes):
    combobox.set(64)
    combobox.grid(row=i+3, column=1, pady=6, padx=7)
    if i != 0:  # Pouze první combobox bude mít readonly stav na začátku
        combobox.state(["disabled"])

dense_layers_scale = tk.Scale(dense_frame, 
                              orient=tk.HORIZONTAL, 
                              from_=0, 
                              to=2, 
                              command=update_dense_scales)

dense_layers_scale.grid(row=2, column=0, rowspan=3, padx=5)
dense_layers_scale.set(1)

lstm_layers_scale.config(command=update_lstm_scales)
dense_layers_scale.config(command=update_dense_scales)



batch_size_label = tk.Label(model_frame, text="Batch size")
batch_size_label.grid(row=3, column=0)

batch_size_combobox = ttk.Combobox(model_frame, 
                                   values=batch_sizes, 
                                   state="readonly",
                                   textvariable=batch_size_var)
batch_size_combobox.set(256)
batch_size_combobox.grid(row=4, column=0, pady=2)


epochs_label = tk.Label(model_frame, text="Total epochs")
epochs_label.grid(row=5, column=0, pady=(20,0))

epochs_scale = tk.Scale(model_frame, 
                         orient=tk.HORIZONTAL, 
                         from_=1, 
                         to=100,
                         length=200)
                         

epochs_scale.set(10)
epochs_scale.grid(row=6, column=0)

run_button = ttk.Button(model_frame,
                        style="Custom2.TButton",
                        text="Run",
                        command=lambda: threading.Thread(target=build_model).start())

run_button.grid(row=7, column=0, pady=15)

save_model_button = ttk.Button(model_frame, 
                               style="Custom2.TButton",
                               text="Save model", 
                               command=lambda: save_model(global_model),
                               state="disabled") 
save_model_button.grid(row=8, column=0, pady=5)


#------------------------------------------------

#-----------------left frame---------------------
# Left frame
left_frame = tk.Frame(root,
                      borderwidth=1, 
                      relief="solid",
                      highlightbackground="#CCCCCC",
                      highlightthickness=1, 
                      bd=0)
left_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

compressed_size_title_label = tk.Label(left_frame, 
                       text="Compressed size",
                       font=("Arial", 12, "bold"))

compressed_size_title_label.grid(row=0, column=0, pady=(5,15))

load_model_button = ttk.Button(left_frame, 
                               style="Custom2.TButton",
                               text="Load model", 
                               command=lambda: load_model())
load_model_button.grid(row=1, column=0, pady=(5,15))

batch_compress_size_label = tk.Label(left_frame, text="Batch size")
batch_compress_size_label.grid(row=2, column=0, pady=(3,0))


batch_compress_size_combobox = ttk.Combobox(left_frame, 
                                   values=batch_compress_sizes, 
                                   state="readonly",
                                   textvariable=batch_compress_size_var)
batch_compress_size_combobox.set(256)
batch_compress_size_combobox.grid(row=3, column=0, pady=(0,5), padx=20)

compressed_size_button = ttk.Button(left_frame,
                                    style="Custom2.TButton",
                                    text="Compute",
                                    command=lambda: threading.Thread
                                    (target=estimated_compressed_size,
                                     args=(text_loader.loaded_text,
                                           context_length_var.get(),
                                           global_model)).start())
compressed_size_button.grid(row=4, column=0, pady=(20, 10))

compressed_size_label = tk.Label(left_frame, textvariable=compressed_size_var)
compressed_size_label.grid(row=5, column=0, pady=5)
#------------------------------------------------

#-----------------bottom frame---------------------
bottom_frame = tk.Frame(root,
                        borderwidth=1, 
                        relief="solid",
                        highlightbackground="#CCCCCC",
                        highlightthickness=1, 
                        bd=0)
bottom_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

bottom_frame.grid_columnconfigure(0, weight=1)

entropy_title_label = tk.Label(bottom_frame, 
                       text="Text information",
                       font=("Arial", 12, "bold"))

entropy_title_label.grid(row=0, column=0, pady=(5,15), sticky="ew")

sigma_label = tk.Label(bottom_frame, textvariable=sigma_var)
sigma_label.grid(row=1, column=0, padx=12, pady=7, sticky="w")

text_size_label = tk.Label(bottom_frame, textvariable=text_size_var)
text_size_label.grid(row=2, column=0, padx=12, pady=7, sticky="w")

entropy_label = tk.Label(bottom_frame, textvariable=kth_entropy_var)
entropy_label.grid(row=3, column=0, padx=12, pady=7, sticky="w")

entropy_button = ttk.Button(bottom_frame, 
                            style="Custom2.TButton",
                            text="Calculate Entropy", 
                            command=lambda: threading.Thread
                            (target=kth_order_entropy, 
                            args=(text_loader.loaded_text, context_length_var.get())).start())
entropy_button.grid(row=4, column=0, pady=(5, 15), padx=22, sticky="w")



#------------------------------------------------

root.mainloop()