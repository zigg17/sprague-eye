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
    and splits each MP4 into frames, storing frames in a folder on the desktop.

    :param file_paths: List of file paths to check and process
    :return: Dictionary with file paths as keys and their corresponding folder paths as values
    """
    # Ensure all files are .mp4
    if not all(Path(file).suffix == '.mp4' for file in file_paths):
        raise ValueError("All files must be in .mp4 format.")

    # Get the desktop path
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    output_dir = os.path.join(desktop_path, "video_frames")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    frame_folders = {}

    for file_path in file_paths:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Create a unique folder for this video inside the desktop directory
        video_name = Path(file_path).stem  # Extract video filename without extension
        video_dir = os.path.join(output_dir, f"frames_{video_name}")
        os.makedirs(video_dir, exist_ok=True)

        frame_folders[file_path] = video_dir

        # Open video with OpenCV
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {file_path}")

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save frame as an image in the directory
            frame_filename = os.path.join(video_dir, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_count += 1
        
        cap.release()
        print(f"Processed {frame_count} frames from {file_path}, stored in {video_dir}")

    return frame_folders.values()

def bbox_folders(model, transform, device, folder_paths):
    for folder in folder_paths:
        print(folder)
        files = [os.path.join(folder, s) for s in os.listdir(folder) if os.path.isfile(os.path.join(folder, s))]
        for _file in files:
            torch_util.predict_and_mask(model, _file, transform, device)
    
    return folder_paths

def images_to_video(image_folder, video_name, frame_rate=30):
    """
    Converts a folder of images into a video.

    Parameters:
    - image_folder (str): Path to the folder containing images.
    - video_name (str): Name of the output video file (default: "output_video.mp4").
    - frame_rate (int): Frames per second for the output video (default: 30).

    Returns:
    - None
    """
    # Get all images and sort them
    images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
    images.sort()  # Ensure they are in order

    if not images:
        print("No images found in the folder.")
        return

    # Read the first image to get dimensions
    first_image_path = os.path.join(image_folder, images[0])
    first_image = cv2.imread(first_image_path)
    
    if first_image is None:
        print("Error loading the first image. Please check the image files.")
        return

    height, width, _ = first_image.shape

    # Define the video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use 'XVID' for .avi
    video = cv2.VideoWriter(video_name, fourcc, frame_rate, (width, height))

    # Loop through images and write to video
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)

        if frame is None:
            print(f"Skipping {img_path}, unable to load.")
            continue

        video.write(frame)

    # Release the video writer
    video.release()
    cv2.destroyAllWindows()

    print(f"Video saved as {video_name}")

def vid_processor(file_path):
    # Get the desktop path
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

    # Define the new directory name
    new_dir = "MyNewFolder"

    # Full path of the new directory
    new_dir_path = os.path.join(desktop_path, new_dir)

    # Create the directory if it doesn't already exist
    if not os.path.exists(new_dir_path):
        os.makedirs(new_dir_path)
    
    

    for folder in file_path:
        last_part = os.path.basename(folder)
        video_name = os.path.join(new_dir_path, (last_part + '.mp4'))
        images_to_video(folder, video_name, frame_rate=30)
        
def convert_image_folders(temp_folders):
    for temp_folder in temp_folders:
        images_to_video(temp_folder, os.path.dirname(temp_folder), )
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