from TextLoader import *
from CustomModel import *
from TrainingProgress import *
from CompressedSize import *
from SequencesGenerator import *
import tkinter as tk
from TopFileFrame import *
from CompressedSizeFrame import *
from ModelFrame import *
from TextInfoFrame import *


class MyApp:
    def __init__(self):
        self.root = None
        self.text_loader = TextLoader()
        self.global_model = None
        self.top_file_frame = None
        self.left_frame = None
        self.model_frame = None
        self.bottom_frame = None

    
    def init_vars(self):
        self.root = tk.Tk()
        self.top_file_frame = TopFileFrame(self)
        self.left_frame = CompressedSizeFrame(self)
        self.model_frame = ModelFrame(self)
        self.bottom_frame = TextInfoFrame(self)
        self.context_length_var = tk.StringVar()
        self.compressed_size_var = tk.StringVar()
        self.kth_entropy_var = tk.StringVar()
        self.batch_compress_size_var = tk.StringVar()
        self.kth_entropy_var.set("Entropy: ")
        self.text_size_var = tk.StringVar()
        self.text_size_var.set("Text size: ")
        self.sigma_var = tk.StringVar()
        self.sigma_var.set("Alphabet size: ")
        self.sigma_value = tk.IntVar()

    def configure_frames(self):
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

    def run(self):
        self.init_vars()
        self.root.title("BP")
        
        self.root.resizable(False, False)
        self.top_file_frame.create_top_file_frame()
        self.model_frame.create_model_frame()
        self.left_frame.create_left_frame()
        self.bottom_frame.create_bottom_frame()
        self.configure_frames()
        self.top_file_frame.create_top_file_frame_widgets()
        self.model_frame.create_model_frame_widgets()
        self.left_frame.create_left_frame_widgets()
        self.bottom_frame.create_bottom_frame_widgets()

        self.root.mainloop()