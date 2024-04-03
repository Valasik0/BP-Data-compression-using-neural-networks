import tkinter as tk
from tkinter import filedialog, messagebox
class TextLoader:
    def __init__(self):
        self.loaded_text = ""
        self.file_name = "No file loaded."

    def load_file(self):
        try:
            file_path = filedialog.askopenfilename()
            if file_path:
                with open(file_path, "rb") as file:
                    self.loaded_text = file.read()
                    self.file_name = file_path
                    messagebox.showinfo("Information", "Text loaded.")
                    
            else:
                self.loaded_text = ""
        except Exception as e:
            messagebox.showerror(title="Error", message=f"An error occurred: {e}")
            self.loaded_text = ""