from tkinter import ttk
import tensorflow as tf
from tensorflow import keras
from TextLoader import *
from CustomModel import *
from KthEntropyCalculator import *
import numpy as np
import threading
import tkinter as tk
from tkinter import filedialog
from collections import defaultdict

#https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

def validate_text(text):
    if not isinstance(text, bytes):
        messagebox.showerror("Error", "Loaded text is in incorrect format.")
        return False
    if len(text) == 0:
        messagebox.showinfo("Info", "Loaded text is empty. Please load a text.")
        return False
    return True

def map_chars(text):
    if text is None:
        return None

    chars = sorted(list(set(text))) #chars - unikatni znaky v textu pomoci metody set
    mapped_chars = {c:i for i, c in enumerate(chars)}

    mapped_chars['<UNK>'] = len(mapped_chars)

    return mapped_chars

def count_unique_chars(mapped_chars): 
    if mapped_chars is None or len(mapped_chars) == 0:
        return 0
    
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
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            return keras.models.load_model(file_path)
        except (OSError, ValueError) as e:
            print(f"Error loading model: {e}")
            return None

def kth_order_entropy(text, k):
    progress_bar.start()
    kth_entropy_calculator = KthEntropyCalculator(text, k)
    entropy = kth_entropy_calculator.calculate_kth_entropy()
    progress_bar.stop()

    entropy_label.config(text=str(round(entropy, 3)))

    return entropy
    
def estimated_compressed_size(text, k, mapped_chars, sigma, model, batch_size=256):
    compressed_size = 0
    batch_sequences = []
    batch_chars = []
    n = len(text)
    epsilon=1e-10

    # předzpracování textu
    mapped_text = [mapped_chars[char] if char in mapped_chars else mapped_chars['<UNK>'] for char in text]

    for i in range(k, len(mapped_text)):
        sequence = mapped_text[i - k:i]
        batch_sequences.append(tf.keras.utils.to_categorical(sequence, num_classes=sigma))
        actual_char_index = mapped_text[i]
        batch_chars.append(actual_char_index)

        if len(batch_sequences) == batch_size or i == len(mapped_text) - 1:
            probabilities = model.predict(np.array(batch_sequences))  #všechny pravděpodobnosti pro následující znak
            p_x = probabilities[np.arange(len(batch_sequences)), batch_chars]  # pravděpodobnosti skutečných následujících znaků
            compressed_size -= np.sum(np.log2(p_x + epsilon))

            batch_sequences = []
            batch_chars = []
    return compressed_size

def update_file_path_label():
    file_path_label.config(text=text_loader.file_name)

def update_lstm_scales(value):
    for combobox in lstm_comboboxes:
        combobox.destroy()
    lstm_comboboxes.clear()

    for i in range(2):
        combobox = ttk.Combobox(lstm_frame, values=nodes_values)
        combobox.grid(row=i+2, column=1, pady=6)
        if i < int(value):
            combobox.state(["readonly"]) 
        else:
            combobox.state(["disabled"])  
        lstm_comboboxes.append(combobox)

def update_dense_scales(value):
    for combobox in dense_comboboxes:
        combobox.destroy()
    dense_comboboxes.clear()

    for i in range(3):
        combobox = ttk.Combobox(dense_frame, values=nodes_values)
        combobox.grid(row=i+2, column=1, pady=6)
        if i < int(value):
            combobox.state(["readonly"]) 
        else:
            combobox.state(["disabled"])  
        dense_comboboxes.append(combobox)

text_loader = TextLoader()

root = tk.Tk()
root.geometry("800x600")

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

left_frame = tk.Frame(root)
left_frame.pack(side=tk.LEFT)

right_frame = tk.Frame(root)
right_frame.pack(side=tk.RIGHT)

top_frame = tk.Frame(left_frame)
top_frame.pack(side=tk.TOP)

bottom_frame = tk.Frame(left_frame)
bottom_frame.pack(side=tk.BOTTOM)

model_frame = tk.Frame(right_frame, 
                       borderwidth=1, 
                       relief="solid",
                       highlightbackground="#CCCCCC",
                       highlightthickness=1, 
                       bd=0)
                       
model_frame.pack(side=tk.RIGHT)

top_file_frame = tk.Frame(root)
top_file_frame.pack(side=tk.TOP, pady=10)

#-----------------top file frame-----------------
file_label = tk.Label(top_file_frame, text="File:")
file_label.pack(side=tk.LEFT, padx=20)

frame = tk.Frame(top_file_frame, 
                 borderwidth=1, 
                 relief="solid", 
                 highlightbackground="#8C8C8C", 
                 highlightthickness=1, 
                 bd=0)

file_path_label = tk.Label(frame, 
                           text=text_loader.file_name, 
                           width=50,
                           anchor="w")

frame.pack(side=tk.LEFT)
file_path_label.pack(side=tk.LEFT)

load_text_button = ttk.Button(top_file_frame, 
                             style="Custom.TButton",
                             text="...", command=lambda: 
                            (text_loader.load_file(),
                            update_file_path_label()))
load_text_button.pack(side=tk.LEFT, padx=5)
#------------------------------------------------

nodes_values = [32, 64, 128]

lstm_frame = tk.Frame(model_frame)
lstm_frame.grid(row=0, column=0, padx=(10,10), pady=(10, 50))
dense_frame = tk.Frame(model_frame)
dense_frame.grid(row=1, column=0, padx=(10,10), pady=(10, 50))

lstm_label = tk.Label(lstm_frame, text="LSTM layers:")
lstm_label.grid(row=0, column=0, columnspan=2)

lstm_number_label = tk.Label(lstm_frame, text="Number")
lstm_number_label.grid(row=1, column=0)

lstm_nodes_label = tk.Label(lstm_frame, text="Nodes")
lstm_nodes_label.grid(row=1, column=1)

lstm_layers_scale = tk.Scale(lstm_frame, orient=tk.HORIZONTAL, from_=0, to=2)
lstm_layers_scale.grid(row=2, column=0, rowspan=2, padx=5)
lstm_layers_scale.set(1)
lstm_comboboxes = [ttk.Combobox(lstm_frame, values=nodes_values) for _ in range(2)]

lstm_comboboxes[0].set(64)
lstm_comboboxes[0].grid(row=3, column=1, pady=6)
lstm_comboboxes[0].state(["disabled"])

lstm_comboboxes[1].grid(row=4, column=1, pady=6)
lstm_comboboxes[1].state(["disabled"])

lstm_layers_scale.config(command=update_lstm_scales)


dense_label = tk.Label(dense_frame, text="Dense layers")
dense_label.grid(row=0, column=0, columnspan=2)

dense_number_label = tk.Label(dense_frame, text="Number")
dense_number_label.grid(row=1, column=0)

dense_nodes_label = tk.Label(dense_frame, text="Nodes")
dense_nodes_label.grid(row=1, column=1)

dense_layers_scale = tk.Scale(dense_frame, orient=tk.HORIZONTAL, from_=1, to=3)
dense_layers_scale.grid(row=2, column=0, rowspan=3, padx=5)
dense_layers_scale.set(1)
dense_comboboxes = [ttk.Combobox(dense_frame, values=nodes_values) for _ in range(3)]

dense_comboboxes[0].set(64)
dense_comboboxes[0].grid(row=3, column=1, pady=6)
dense_comboboxes[0].state(["disabled"])

dense_comboboxes[1].grid(row=4, column=1, pady=6)
dense_comboboxes[1].state(["disabled"])

dense_comboboxes[2].grid(row=5, column=1, pady=6)
dense_comboboxes[2].state(["disabled"])



dense_layers_scale.config(command=update_dense_scales)

run_button = ttk.Button(model_frame,
                        style="Custom2.TButton",
                        text="Run")
run_button.grid(row=2, column=0)



context_length_label = tk.Label(top_frame, text="Délka kontextu")
context_length_label.pack(padx=5)

context_length_combobox = ttk.Combobox(top_frame, 
                                       textvariable=context_length_var, 
                                       values=context_lengths, 
                                       state="readonly")
context_length_combobox.pack(padx=5)
context_length_combobox.current(3)


#tk.Button(root, text="Save model", command=lambda: save_model(custom_model)).pack()
load_model_button = ttk.Button(bottom_frame, 
                               style="Custom2.TButton",
                               text="Load model", 
                               command=lambda: load_model())
load_model_button.pack()
entropy_button = ttk.Button(bottom_frame, 
                            style="Custom2.TButton",
                            text="Kth order entropy", 
                            command=lambda: threading.Thread
                            (target=kth_order_entropy, 
                            args=(text_loader.loaded_text, int(context_length_var.get()))).start())
entropy_button.pack()

entropy_label = tk.Label(root, text="")
entropy_label.pack()

progress_var = tk.IntVar()
progress_bar = ttk.Progressbar(bottom_frame, length=150, mode='indeterminate')
progress_bar.pack()

root.mainloop()


"""
mapped_chars = map_chars(text_loader.loaded_text)
sigma = count_unique_chars(mapped_chars) #pocet unikatnich znaku
train_data_generator = generate_sequences(text_loader.loaded_text, 3, mapped_chars, sigma)

num_dense_layers = 1
dense_layer_sizes = [100]
dense_activations = ['relu']

custom_model = CustomModel(io_size=sigma, k=3, num_dense_layers=num_dense_layers, dense_layer_sizes=dense_layer_sizes, dense_activations=dense_activations)
#opt = tf.keras.optimizers.Adam()
custom_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
custom_model.fit(train_data_generator, epochs=1, steps_per_epoch=len(text_loader.loaded_text)-3, callbacks=[early_stopping])
"""