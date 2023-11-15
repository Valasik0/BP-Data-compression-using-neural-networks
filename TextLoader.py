import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import *
class TextLoader:
    def __init__(self):
        self.loaded_text = ""
        self.file_name = ""

    def load_file(self):
        try:
            file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
            if file_path:
                with open(file_path, "r") as file:
                    self.loaded_text = file.read()
                    self.file_name = file_path.split("/")[-1]
                    messagebox.showinfo("Information", "Text loaded.")
            else:
                self.loaded_text = ""
        except Exception as e:
            messagebox.showerror(title="Error", message=f"An error occurred: {e}")
            self.loaded_text = ""