from tkinter import messagebox
class TextAnalyzer:
    def __init__(self, text):
        self.text = text
        self.is_valid = self.validate_text() 
        self.mapped_chars = self.compute_mapped_chars()
        self.sigma = self.compute_unique_chars()
        

    def compute_unique_chars(self):
        if not self.is_valid:
            return None
        
        return len(self.mapped_chars)

    def compute_mapped_chars(self):
        if self.is_valid: 
            chars = sorted(list(set(self.text)))  # chars - unikatni znaky v textu pomoci metody set
            mapped_chars = {c: i for i, c in enumerate(chars)}

            mapped_chars['<UNK>'] = len(mapped_chars)
            return mapped_chars
        
        else:
            return None

    def validate_text(self):
        if not isinstance(self.text, bytes):
            messagebox.showerror("Error", "Loaded text is in incorrect format.")
            return False
        if len(self.text) == 0:
            messagebox.showinfo("Info", "Loaded text is empty. Please load a text.")
            return False
        return True