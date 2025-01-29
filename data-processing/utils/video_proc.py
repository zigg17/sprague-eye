import os
import tempfile
from pathlib import Path
import shutil

import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import ffmpeg

import utils.pytorch as torch_util

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
        files = [os.path.join(directory, s) for s in files]
        return directory, files
    else:
        messagebox.showerror(
                "Invalid Files",
                "The selected directory contains non-MP4 files. Please select a folder with only MP4 files."
            )
        return None  

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

def convert_video_folders(model, transform, device, folder_paths):
    for folder in folder_paths:
        file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
        for _file in file_paths:
            predict_and_mask(model, _file, transform, device)
    
    return folder_paths

def images_to_video(image_folder: str, video_name: str, fps: int = 30, image_ext: str = "jpg"):
    """
    Converts a folder of images into an MP4 video and saves it in a new folder on the Desktop.

    Args:
        image_folder (str): Path to the folder containing images.
        video_name (str): Name of the output video file (e.g., "output.mp4").
        fps (int, optional): Frames per second. Defaults to 30.
        image_ext (str, optional): Image extension (e.g., "jpg", "png"). Defaults to "jpg".
    
    Returns:
        None
    """
    # Get Desktop path
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

    # Create output directory if it doesn't exist
    output_folder = os.path.join(desktop_path, "Generated_Videos")
    os.makedirs(output_folder, exist_ok=True)

    # Full output video path
    output_video = os.path.join(output_folder, video_name)

    # Ensure images are sorted and correctly named
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(f'.{image_ext}')])
    if not images:
        raise ValueError(f"No images with extension .{image_ext} found in {image_folder}")

    # Create an input pattern (e.g., frame%d.jpg)
    input_pattern = os.path.join(image_folder, images[0]).replace(images[0], f"%d.{image_ext}")

    # Run FFmpeg processing
    (
        ffmpeg
        .input(input_pattern, framerate=fps)
        .output(output_video, vcodec='libx264', pix_fmt='yuv420p')
        .run()
    )
    
def convert_image_folders(temp_folders):
    # test
    return

def cleanup_temp_folders(temp_folders):
    """
    Deletes the temporary folders created for storing video frames.

    :param temp_folders: Dictionary of video file paths to temporary folder paths.
    """
    for folder in temp_folders.values():
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"Deleted temporary folder: {folder}")

if __name__ == "__main__":
    selected_directory, files = get_directory()

    folders = process_videos_to_frames(files)

    cleanup_temp_folders(folders)