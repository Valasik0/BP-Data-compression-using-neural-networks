import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import MinMaxScaler
from TextLoader import *
from CustomModel import *
import numpy as np
#https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

def map_chars(text):
    if text is None:
        messagebox.showinfo("Information", "The text is empty. Please load a file.")
        return None

    chars = sorted(list(set(text)))
    mapped_chars = dict((c,i) for i,c in enumerate(chars))

    return mapped_chars

def count_unique_chars(mapped_chars): 
    if mapped_chars is None or len(mapped_chars) == 0:
        return 0
    
    return len(mapped_chars)

def create_training_data(text, k, mapped_chars, sigma):
    sequences = []

    for i in range(k, len(text)):
        sequence = text[i-k:i+1]
        mapped_sequence = [mapped_chars[char] for char in sequence]
        sequences.append(mapped_sequence)

    sequences = np.array(sequences)
    train_chars, train_labels = sequences[:,:-1], sequences[:,-1]

    onehot_encoded = [tf.keras.utils.to_categorical(x,num_classes=sigma) for x in train_labels]

    train_chars = tf.keras.utils.to_categorical(train_chars,num_classes=sigma)
    train_labels = np.array(onehot_encoded)

    return train_chars, train_labels 

def run_model(train_chars, train_labels):
    model = keras.Sequential()
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(128, activation="softmax"))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_chars, train_labels, epochs=5)

root = tk.Tk()
root.geometry("350x500")
root.title("BP")

num_layers_var = IntVar() 
activation_var = StringVar()



activation_functions = ["relu", "sigmoid", "tanh", "logistic", "identity"]

layers_slider = tk.Scale(root, 
                         from_=1, 
                         to=5, 
                         variable=num_layers_var, 
                         orient="horizontal", 
                         tickinterval=1, 
                         showvalue=True, 
                         length=200,
                         label="Number of layers")
layers_slider.set(3)
layers_slider.pack(pady=10)


activation_combobox = ttk.Combobox(root, textvariable=activation_var, values=activation_functions)
activation_combobox.set(activation_functions[0])
activation_combobox.pack(padx=10, pady=5)

text_loader = TextLoader()
text_loader.load_file()

open_button = ttk.Button(root, text="Open Text File", command=text_loader.load_file)
open_button.pack(padx=10, pady=10)

label = tk.Label(root, text=text_loader.file_name)
label.pack()

mapped_chars = map_chars(text_loader.loaded_text)
sigma = count_unique_chars(mapped_chars) #pocet unikatnich znaku
train_chars, train_labels = create_training_data(text_loader.loaded_text, 3, mapped_chars, sigma)

custom_model = CustomModel(io_size=sigma, k=3)
custom_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
custom_model.fit(train_chars, train_labels, epochs=10)

#train_button = ttk.Button(root, text="Run", command=lambda: run_model(train_chars, train_labels))
#train_button.pack(padx=10, pady=10)

root.mainloop()

