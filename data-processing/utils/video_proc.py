import os
import tkinter as tk
from tkinter import filedialog, messagebox
import tempfile
from pathlib import Path
import cv2

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
        files = [directory + s for s in files]
        return directory, files
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

def process_videos_to_frames(file_paths):
    """
    Takes a list of file paths, checks if they're all MP4s,
    and splits each MP4 into frames, storing frames in temporary folders.

    :param file_paths: List of file paths to check and process
    :return: Dictionary with file paths as keys and their corresponding temp folder paths as values
    """
    # Check if all files are .mp4
    if not all(Path(file).suffix == '.mp4' for file in file_paths):
        raise ValueError("All files must be in .mp4 format.")
    
    temp_folders = {}

    for file_path in file_paths:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Create a temporary directory for the video
        temp_dir = tempfile.mkdtemp(prefix="frames_")
        temp_folders[file_path] = temp_dir

        # Open video with OpenCV
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {file_path}")

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save frame as an image in the temporary directory
            frame_filename = os.path.join(temp_dir, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_count += 1
        
        cap.release()
        print(f"Processed {frame_count} frames from {file_path}, stored in {temp_dir}")

    return temp_folders
    