import tkinter as tk
from tkinter import ttk, filedialog, messagebox
class TextLoader:
    def __init__(self):
        self.loaded_text = ""

    def load_file(self):
        try:
            file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
            if file_path:
                with open(file_path, "r", encoding='utf-8') as file:
                    self.loaded_text = file.read()
            else:
                self.loaded_text = ""
        except Exception as e:
            messagebox.showerror(title="Error", message=f"An error occurred: {e}")
            self.loaded_text = ""