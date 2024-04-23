import os
import tkinter as tk
from tkinter import filedialog, messagebox
class TextLoader:
    def __init__(self):
        self.loaded_text = ""
        self.file_name = "No file loaded."
        self.file_size = ""

    def load_file(self):
        try:
            file_path = filedialog.askopenfilename()
            if file_path:
                with open(file_path, "rb") as file:
                    self.loaded_text = file.read()
                    self.file_name = file_path
                    self.file_size = self.format_file_size(file_path)
                    return True
            else:
                return None
        except Exception as e:
            messagebox.showerror(title="Error", message=f"An error occurred: {e}")
            self.loaded_text = ""

    def format_file_size(self, file_path):
        units = [("B", 1), ("KB", 1024), ("MB", 1024**2), ("GB", 1024**3), ("TB", 1024**4)]
        size_in_bytes = os.path.getsize(file_path)

        for unit, limit in reversed(units):
            if size_in_bytes >= limit:
                return f"{size_in_bytes / limit:.2f} {unit}"
        
        return "0 B"