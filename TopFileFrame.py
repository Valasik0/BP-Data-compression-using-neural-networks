from tkinter import ttk
from TextLoader import *
from CustomModel import *
from TrainingProgress import *
from CompressedSize import *
from SequencesGenerator import *
import tkinter as tk


class TopFileFrame:
    def __init__(self, app):
        self.top_file_frame = None
        self.file_label = None
        self.app = app	
        self.frame = None
        self.file_label = None
        self.load_text_button = None
        self.context_length_label = None
        self.context_length_combobox = None
        self.context_lengths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]

    def create_top_file_frame(self):
            self.top_file_frame = tk.Frame(self.app.root,
                                        borderwidth=1, 
                                        relief="solid",
                                        highlightbackground="#CCCCCC",
                                        highlightthickness=1, 
                                        bd=0)
            self.top_file_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew", columnspan=2)

    def create_top_file_frame_widgets(self):
        self.file_label = tk.Label(self.top_file_frame, text="File: ")
        self.file_label.grid(row=0, column=0, padx=(20, 20), sticky="w", pady=4)

        self.frame = tk.Frame(self.top_file_frame, 
                        borderwidth=1, 
                        relief="solid",
                        highlightbackground="#CCCCCC",
                        highlightthickness=1, 
                        bd=0,)

        self.file_path_label = tk.Label(self.frame, 
                                text=self.app.text_loader.file_name, 
                                width=45,
                                anchor="w",
                                pady=4)

        self.frame.grid(row=0, column=1, padx=20, sticky="w")
        self.file_path_label.pack(side=tk.LEFT)

        self.load_text_button = ttk.Button(self.top_file_frame, 
                                    style="Custom.TButton",
                                    text="...", 
                                    command=lambda: (self.app.text_loader.load_file(), 
                                                    self.update_file_path_label()),
                                    width=5) 
        self.load_text_button.grid(row=0, column=2, padx=5, pady=4, sticky="w")


        self.context_length_label = tk.Label(self.top_file_frame, text="Context length:")
        self.context_length_label.grid(row=1, column=0, padx=20) 

        self.context_length_combobox = ttk.Combobox(self.top_file_frame, 
                                            textvariable=self.app.context_length_var, 
                                            values=self.context_lengths, 
                                            state="readonly",
                                            width=10)
        self.context_length_combobox.grid(row=1, column=1, pady=5, padx=(20, 0), sticky="w")

    def update_file_path_label(self):
        ta = TextAnalyzer(self.app.text_loader.loaded_text)
        self.app.sigma_value.set(ta.sigma)
        self.app.text_size_var.set(f"Text size: {self.app.text_loader.file_size}")
        self.app.sigma_var.set(f"Alphabet size: {self.app.sigma_value.get()}")
        self.file_path_label.config(text=self.app.text_loader.file_name)
        if self.app.model_frame.tree.exists(self.app.model_frame.output_layer_item):
            self.app.model_frame.tree.set(self.app.model_frame.output_layer_item, column=2, 
                                        value="Alphabet size (" + str(self.app.sigma_value.get()) + ")")
        self.app.kth_entropy_var.set("Entropy: ")
        
    def load_file(self):
        self.app.text_loader.load_file()
        self.update_file_path_label()