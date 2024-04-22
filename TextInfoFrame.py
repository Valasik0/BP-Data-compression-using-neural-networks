from tkinter import ttk
from TextLoader import *
from CustomModel import *
from TrainingProgress import *
from CompressedSize import *
from SequencesGenerator import *
import threading
import tkinter as tk
from TopFileFrame import *
import time
import entropy

class TextInfoFrame:
    def __init__(self, app):
        self.app = app
        self.bottom_frame = None
        self.entropy_title_label = None
        self.sigma_label = None
        self.text_size_label = None
        self.entropy_label = None
        self.entropy_button = None
        self.result = None


    def create_bottom_frame(self):
        self.bottom_frame = tk.Frame(self.app.root,
                        borderwidth=1, 
                        relief="solid",
                        highlightbackground="#CCCCCC",
                        highlightthickness=1, 
                                bd=0)
        self.bottom_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")


    def create_bottom_frame_widgets(self):
        self.bottom_frame.grid_columnconfigure(0, weight=1)

        self.entropy_title_label = tk.Label(self.bottom_frame, 
                            text="Text information",
                            font=("Arial", 12, "bold"))

        self.entropy_title_label.grid(row=0, column=0, pady=(5,15), sticky="ew")

        self.sigma_label = tk.Label(self.bottom_frame, textvariable=self.app.sigma_var)
        self.sigma_label.grid(row=1, column=0, padx=12, pady=7, sticky="w")

        self.text_size_label = tk.Label(self.bottom_frame, textvariable=self.app.text_size_var)
        self.text_size_label.grid(row=2, column=0, padx=12, pady=7, sticky="w")

        self.entropy_label = tk.Label(self.bottom_frame, textvariable=self.app.kth_entropy_var)
        self.entropy_label.grid(row=3, column=0, padx=12, pady=7, sticky="w")

        self.entropy_button = ttk.Button(self.bottom_frame, 
                                    style="Custom2.TButton",
                                    text="Calculate Entropy", 
                                    command=lambda: threading.Thread
                                    (target=self.kth_order_entropy, 
                                    args=(self.app.text_loader.loaded_text, self.app.context_length_var.get())).start())
        self.entropy_button.grid(row=4, column=0, pady=(5, 15), padx=22, sticky="w")

    def kth_order_entropy(self, text, k):
        ta = TextAnalyzer(text)
        if not ta.validate_text():
            return
        
        if not k:
            tk.messagebox.showinfo("Info", "No context lenght selected")
            return

        self.app.kth_entropy_var.set("Entropy: calculating...")


        start_time1 = time.time()
        entropy_thread = threading.Thread(target=self.thread_calculate_entropy, args=(text, k))
        entropy_thread.start()
        entropy_thread.join()
        end_time1 = time.time()


        print(f"Entropy calculation time: {end_time1 - start_time1} seconds")

        self.entropy_label.config(text=str(round(self.result, 3)))

        self.app.kth_entropy_var.set(f"Entropy: {round(self.result, 3)} bpB")

    def thread_calculate_entropy(self, text, k):
        entropy_calculator = entropy.KthEntropyCalculator(text, int(k))
        self.result = entropy_calculator.calculate_kth_entropy()