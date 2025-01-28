import os
import tkinter as tk
from tkinter import filedialog, messagebox

def get_directory():
    root = tk.Tk()
    root.withdraw() 

    directory = filedialog.askdirectory(title="Select a Directory")

    if not directory:
        messagebox.showerror("No Directory Selected", "You must select a directory.")
        return None

    files = [
        f for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and not f.startswith('.')
    ]

    if not files:
            messagebox.showerror("Empty Directory", "The selected directory is empty. Please select a valid folder.")
            return None

    if all(f.endswith('.mp4') for f in files if os.path.isfile(os.path.join(directory, f))):
        return directory 
    else:
        messagebox.showerror(
                "Invalid Files",
                "The selected directory contains non-MP4 files. Please select a folder with only MP4 files."
            )
        return None  

if __name__ == "__main__":
    selected_directory = get_directory()
    if selected_directory:
        print(f"You selected: {selected_directory}")
    else:
        print("No directory was selected.")