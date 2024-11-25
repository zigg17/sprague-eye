from tkinter import filedialog
from tkinter import messagebox
import customtkinter as CTk
from customtkinter import CTkImage
from PIL import Image, ImageTk
from collections import defaultdict
import os
import shutil
import sys
import cv2
import time
import csv
import re
from datetime import datetime
import pandas as pd

def resource_path(relative_path):
        """ Get absolute path to resource, works for dev and for PyInstaller """
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_path, relative_path)


class Application(CTk.CTk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.iconpath = ImageTk.PhotoImage(file=resource_path(os.path.join("images", "ratIcon.ico")))

        self.wm_iconbitmap()
        self.iconphoto(False, self.iconpath)
        
        # Initialize app settings
        CTk.set_appearance_mode('dark')
        self.title("Rat Swipe")
        self.position_window(800, 550)
        self.resizable(False, False)
        
        self.grid_rowconfigure(0, weight=1)  # Content area
        self.grid_rowconfigure(1, weight=0)  # Navigation area
        self.grid_columnconfigure(0, weight=1)

        self.directory_path = os.path.join(os.path.expanduser('~'), 'weaveData','userData.txt')
        
        self.spliceFrame = SpliceFrame(self)
        self.statsFrame = StatsFrame(self)

        self.isOpen = False


        self.swipeFrame = SwipeFrame(self)
        self.navFrame = NavigationFrame(self)
        self.navFrame.grid(row=1, column=0, sticky="ew")  # Bottom-aligned navbar

        self.spliceFrame.grid(row=0, column=0, sticky="nsew")
        self.navFrame.splice.configure(fg_color=("#ff5146", "#ff5146"))
        
    def select_frame_by_name(self, name):
        self.navFrame.update_button_color(name)

        # Grid management for frames
        if name == "splice":
            self.spliceFrame.grid(row=0, column=0, sticky="nsew")
        else:
            self.spliceFrame.grid_forget()

        if name == "swipe":
            self.swipeFrame.grid(row=0, column=0, sticky="nsew")
        else:
            self.swipeFrame.grid_forget()

        if name == "stats":
            self.statsFrame.grid(row=0, column=0, sticky="nsew")
        else:
            self.statsFrame.grid_forget()
        
    def position_window(self, width, height):
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width/2) - (width/2)
        y = (screen_height/2) - (height/2)
        self.geometry('%dx%d+%d+%d' % (width, height, x, y))

class NavigationFrame(CTk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, corner_radius=0)
        self.parent = parent
        self.grid_rowconfigure(9, weight=1)
        self.create_widgets()
    
    def update_button_color(self, name):
        # set button color for selected button
        self.splice.configure(fg_color=("#ff5146", "#ff5146") if name == "splice" else "transparent")
        self.swipe.configure(fg_color=("#ff5146", "#ff5146") if name == "swipe" else "transparent")
        self.stats.configure(fg_color=("#ff5146", "#ff5146") if name == "stats" else "transparent")

    def create_widgets(self):
        total_columns = 10

        # Splice button
        self.splice = CTk.CTkButton(self, corner_radius=0, height=40, border_spacing=10, text="Splice", width=270,
                                    fg_color="transparent", text_color=("gray10", "gray90"), 
                                    hover_color=("#ff6961", "#ff6961"), anchor="center", command=self.on_splice_click)
        self.splice.grid(row=0, column=3, columnspan=1, sticky="ew")

        # Swipe button
        self.swipe = CTk.CTkButton(self, corner_radius=0, height=40, border_spacing=10, text="Swipe", width=270,
                                fg_color="transparent", text_color=("gray10", "gray90"), 
                                hover_color=("#ff6961", "#ff6961"), anchor="center", command=self.on_meditation_click)
        self.swipe.grid(row=0, column=4, columnspan=1, sticky="ew")

        # Stats button
        self.stats = CTk.CTkButton(self, corner_radius=0, height=40, border_spacing=10, text="Stats", width=270,
                                fg_color="transparent", text_color=("gray10", "gray90"), 
                                hover_color=("#ff6961", "#ff6961"), anchor="center", command=self.on_journal_click)
        self.stats.grid(row=0, column=5, columnspan=1, sticky="ew")

        # Setting empty columns on either side for spacing
        for i in range(3):
            self.grid_columnconfigure(i, weight=1)
        for i in range(7, total_columns):
            self.grid_columnconfigure(i, weight=1)
    
    def on_splice_click(self):
        self.parent.select_frame_by_name("splice")
    
    def on_meditation_click(self):
        self.parent.select_frame_by_name("swipe")
    
    def on_journal_click(self):
        self.parent.select_frame_by_name("stats")

class FolderNameDialog(CTk.CTkToplevel):
    def __init__(self, parent, selected_folder, title=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.selected_folder = selected_folder

        if title:
            self.title(title)
        self.position_window(300,150)

        CTk.set_appearance_mode("dark")
        
        # Add customtkinter widgets
        self.label = CTk.CTkLabel(self, text="Enter folder name:")
        self.label.pack(pady=10)

        self.entry = CTk.CTkEntry(self)
        self.entry.pack(pady=10)

        self.submit_button = CTk.CTkButton(self, text="Create",
                                           fg_color=("gray75", "gray30"),  # Custom colors
                                           hover_color=("gray30", "gray75"),
                                           command=self.on_submit)
        self.submit_button.pack(pady=10)

        self.folder_name = None
        self.new_folder_path = None
    
    def position_window(self, width, height):
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width/2) - (width/2)
        y = (screen_height/2) - (height/2)
        self.geometry('%dx%d+%d+%d' % (width, height, x, y))

    def on_submit(self):
        self.folder_name = self.entry.get()

        if not self.folder_name:
            messagebox.showwarning("Warning", "Folder name cannot be empty.")
            return

        # Add check to ensure selected_folder is not None
        if self.selected_folder is None:
            messagebox.showwarning("Warning", "No folder selected.")
            return

        if self.folder_name == os.path.basename(self.selected_folder):
            messagebox.showwarning("Warning", "Cannot be the same name as the selected folder.")
            return

        if os.path.exists(os.path.join(os.path.expanduser("~"), "Desktop", self.folder_name)):
            messagebox.showwarning("Warning", "Choose a name of a folder that doesn't exist.")
            return

        self.destroy()

    def show(self):
        self.wait_window()
        return self.folder_name

class FolderNameDialog_swipe(CTk.CTkToplevel):
    def __init__(self, parent, title=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        if title:
            self.title(title)
        self.position_window(300,150)

        CTk.set_appearance_mode("dark")
        
        # Add customtkinter widgets
        self.label = CTk.CTkLabel(self, text="Enter folder name:")
        self.label.pack(pady=10)

        self.entry = CTk.CTkEntry(self)
        self.entry.pack(pady=10)

        self.submit_button = CTk.CTkButton(self, text="Create",
                                           fg_color=("gray75", "gray30"),  # Custom colors
                                           hover_color=("gray30", "gray75"),
                                           command=self.on_submit)
        self.submit_button.pack(pady=10)

        self.folder_name = None
        self.new_folder_path = None
    
    def position_window(self, width, height):
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width/2) - (width/2)
        y = (screen_height/2) - (height/2)
        self.geometry('%dx%d+%d+%d' % (width, height, x, y))

    def on_submit(self):
        self.folder_name = self.entry.get()

        if not self.folder_name:
            messagebox.showwarning("Warning", "Folder name cannot be empty.")
            return

        self.desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
        self.new_folder_path = os.path.join(self.desktop_path, self.folder_name)

        if os.path.exists(self.new_folder_path):
            messagebox.showwarning("Warning", "Choose a name of a folder that doesn't exist.")
            return

        try:
            # Attempt to create the folder and catch any errors
            os.makedirs(self.new_folder_path, exist_ok=False)
            messagebox.showinfo("Success", f"Folder '{self.folder_name}' created successfully.")
        except OSError as error:
            # Handle the error if folder creation fails
            messagebox.showerror("Error", f"Folder creation was unsuccessful: {error}")
            return

        self.destroy()  # Only destroy after folder creation is successful

    def show(self):
        self.wait_window()
        return self.folder_name

class ClassCountDialog(CTk.CTkToplevel): 
    def __init__(self, parent, title=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        if title:
            self.title(title)
        self.position_window(500,150)
        
        CTk.set_appearance_mode("dark")
        
        self.label = CTk.CTkLabel(self, text="Type classes separated by comma, i.e. explore,avoid,escape")
        self.label.pack(pady=10)

        self.entry = CTk.CTkEntry(self)
        self.entry.pack(pady=10)

        self.submit_button = CTk.CTkButton(self, text="Enter",
                                           fg_color=("gray75", "gray30"),
                                           hover_color=("gray30", "gray75"),
                                           command=self.on_submit)
        self.submit_button.pack(pady=10)

        self.class_string = None

    def position_window(self, width, height):
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width/2) - (width/2)
        y = (screen_height/2) - (height/2)
        self.geometry('%dx%d+%d+%d' % (width, height, x, y))

    def on_submit(self):
        self.class_string = self.entry.get()

        if not self.class_string:
            messagebox.showwarning("Warning", "Entry cannot be empty.")
            return
        
        pattern = r"^[a-zA-Z0-9]+(,[a-zA-Z0-9]+)*$"

        if not re.match(pattern, self.class_string):
            messagebox.showwarning("Warning", "Invalid format. Please enter words separated by commas, e.g., explore,avoid,escape.")
            return

        self.class_list = self.class_string.split(",")
        self.class_count = len(self.class_list)
        print(self.class_count)
        print(self.class_list)

        if self.class_count > 4:
            messagebox.showwarning("Warning", "Please enter no more than 4 classes.")
            return

        self.destroy()

    def show(self):
        self.wait_window()
        return self.class_string


class SpliceFrame(CTk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, corner_radius=0, fg_color="transparent")
        self.parent = parent
        self.width = 760  # Adjust the width as needed
        self.height = 100  # Adjust the height as needed
        self.configure(width=self.width, height=self.height)

        self.step2_check = False
        self.step3_check = False
        self.step4_check = False
        
        # Label 1
        self.label1 = CTk.CTkLabel(self, text="1) Select folder for processing:")
        self.label1.pack(pady=15)
        # Load the green checkmark image
        self.checkmark_image = CTk.CTkImage(dark_image =Image.open(resource_path(os.path.join("images", "green_checkmark.png"))),
                                       light_image =Image.open(resource_path(os.path.join("images", "green_checkmark.png"))), 
                                    size=(20, 20))

        # Button for Select File
        self.select_folder_button = CTk.CTkButton(self, text="Select folder", command=self.select_folder, 
                                                hover_color=("#ff6961", "#ff6961"), fg_color="#3a3e41")
        self.select_folder_button.pack(pady=5)

        # Label 2
        self.label2 = CTk.CTkLabel(self, text="2) Create folder to store labeled images:")
        self.label2.pack(pady=15)

        # Button for Create Folder
        self.create_folder_button = CTk.CTkButton(self, text="Create Folder", command=self.create_folder,
                                                  hover_color=("#ff6961", "#ff6961"), fg_color="#3a3e41")
        self.create_folder_button.pack(pady=5)

        # Label 3
        self.label3 = CTk.CTkLabel(self, text="3) Choose number of classes")
        self.label3.pack(pady=15)

        # Button for Splice Videos
        self.quantify = CTk.CTkButton(self, text="Class Entry", command=self.quantify_class,
                                                  hover_color=("#ff6961", "#ff6961"), fg_color="#3a3e41")
        self.quantify.pack(pady=5)

        # Label 4
        self.label3 = CTk.CTkLabel(self, text="4) Splice videos for labelling:")
        self.label3.pack(pady=15)

        # Button for Splice Videos
        self.splice_videos_button = CTk.CTkButton(self, text="Splice Videos", command=self.splice_videos,
                                                  hover_color=("#ff6961", "#ff6961"), fg_color="#3a3e41")
        self.splice_videos_button.pack(pady=5)

        self.load_label = CTk.CTkLabel(self, text="X) Click to load previous work:")
        self.load_label.pack(pady = 15)  # Adjust x and y coordinates as nee
        
        # Create labels and buttons
        self.loading_button = CTk.CTkButton(self, text="Load Previous", command = self.load_previous_data,
                                            hover_color=("#ff6961", "#ff6961"), fg_color="#3a3e41")
        self.loading_button.pack(pady = 5)  # Adjust x and y coordinates as nee

        self.progress_bar = CTk.CTkProgressBar(self, determinate_speed = 3, indeterminate_speed = 3,
                                               progress_color = '#ff6961')
        self.progress_bar.pack(pady=5)
        self.progress_bar.set(0)
        self.progress_bar.pack_forget()  # Initially hide the progress bar

        self.new_folder_path = None
        self.selected_folder = None

    def select_folder(self):
        # Open the folder selection dialog
        folder_path = filedialog.askdirectory(title="Select a folder")

        # Check if a folder was selected
        if folder_path:
            # Check all files in the folder for their extensions
            all_files_are_videos = all(file.lower().endswith(('.mp4', '.mov')) for file in os.listdir(folder_path))

            if not all_files_are_videos:
                # If there are files that are not videos, show a message box and exit the function
                messagebox.showerror("Error", "Please select a folder with videos only (.mp4, .mov).")
                return

            self.select_folder_button.configure(self, text="Select folder",
                                                hover_color=("#ff6961", "#ff6961"), fg_color="#3a3e41", 
                                                image=self.checkmark_image)
            self.selected_folder = folder_path
        else:
            # No folder was selected
            messagebox.showerror("Error", "No folder was selected.")

    def create_folder(self):
        # Check if a source folder has been selected before proceeding
        if self.selected_folder is None:
            messagebox.showerror("Error", "Please select a source folder first.")
            return

        dialog = FolderNameDialog(self, self.selected_folder, title="Create New Folder")
        folder_name = dialog.show()

        # Validate the folder name
        if folder_name:
            self.desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
            self.new_folder_path = os.path.join(self.desktop_path, folder_name)

            try:
                # Attempt to create the folder and catch any errors
                os.makedirs(self.new_folder_path, exist_ok=False)
                messagebox.showinfo("Success", f"Folder '{folder_name}' created successfully.")
                
                # Update the button UI to reflect the successful folder creation
                self.create_folder_button.configure(text="Folder Created", 
                                                    hover_color=("#ff6961", "#ff6961"), 
                                                    fg_color="#3a3e41", 
                                                    image=self.checkmark_image)
            except OSError as error:
                # Handle the error if folder creation fails
                messagebox.showerror(f"{error}", "Folder creation was unsuccessful.")
        else:
            messagebox.showerror("Error", "Folder creation was canceled.")

    def quantify_class(self):
        # Open the dialog to get the class quantification input
        dialog = ClassCountDialog(self, title="Class Quantification")
        class_string = dialog.show()  # Get the class string from the dialog

        if class_string:
            class_list = class_string.split(",")
            class_count = len(class_list)
            
            if self.new_folder_path:
                class_file_path = os.path.join(self.new_folder_path, 'class.txt')
                
                try:
                    with open(class_file_path, 'w') as file:
                        file.write(f"Class count: {class_count}\n")
                        file.write(f"Class list: {', '.join(class_list)}\n")
                        
                    messagebox.showinfo("Success", "Class information has been saved to class.txt successfully.")

                except Exception as error:
                    messagebox.showerror("Error", f"Failed to write to class.txt: {error}")

            self.quantify.configure(
                text="Class Entry",
                hover_color=("#ff6961", "#ff6961"),
                fg_color="#3a3e41",
                image=self.checkmark_image
            )
        else:
            messagebox.showinfo("Information", "Class quantification was canceled.")


    def splice_videos(self):
        # Check if the folders are set
        if self.selected_folder == None or self.new_folder_path == None:
            messagebox.showerror("Error", "Please select the source and destination folders first.")

            return
        
        self.load_label.forget()
        self.loading_button.forget()
        self.progress_bar.pack(pady = 50)
        
        
        # Process each video in the selected folder
        for filename in os.listdir(self.selected_folder):
            if filename.lower().endswith(('.mp4', '.mov')):
                video_path = os.path.join(self.selected_folder, filename)
                self.process_video(video_path)           

        self.progress_bar.set(1)

        csv_file_path = os.path.join(self.new_folder_path, 'labels.csv')

        with open(csv_file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Image Path', 'Class Label'])  # Add column names

        current_datetime = datetime.now()
        datetime_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        txt_file_path = os.path.join(self.new_folder_path, 'info.txt')

        
        with open(txt_file_path, 'w') as file:
            file.write('Intialized @ ' + datetime_string + '\n0')
        
        self.parent.isOpen = True

        messagebox.showinfo("Success", "All videos have been processed.")
        self.splice_videos_button.configure(self, text="Splice Videos", 
                                                hover_color=("#ff6961", "#ff6961"), fg_color="#3a3e41", 
                                                image=self.checkmark_image)
        self.parent.swipeFrame.load_images()
        self.parent.swipeFrame.display_image()

    def load_previous_data(self):
        # Open the folder selection dialog
        folder_path = filedialog.askdirectory(title="Select a folder")

        # Check if a folder was selected
        if folder_path:
            # Check for the required files
            required_files = {'labels.csv', 'info.txt', 'class.txt', 'image_paths.txt'}
            folder_files = set(os.listdir(folder_path))
            
            # Check if all required files are present
            if required_files.issubset(folder_files):
                # Check if all other files are .jpg files
                for file in folder_files - required_files:  # Files other than required ones
                    if not file.lower().endswith('.jpg'):
                        messagebox.showerror("Error", "The folder does not contain only .jpg files.")
                        return
                
                # If everything is fine, proceed with loading
                self.new_folder_path = folder_path
                # Call the functions you want to execute after loading
                self.parent.isOpen = True
                self.new_folder_path = folder_path
                self.parent.swipeFrame.load_images()  # Assuming these methods exist
                self.parent.swipeFrame.display_image()
                self.select_folder_button.configure(self, text="Select folder",
                                                    hover_color=("#ff6961", "#ff6961"), fg_color="#3a3e41", 
                                                    image=self.checkmark_image)
                self.create_folder_button.configure(self, text="Create Folder",
                                                    hover_color=("#ff6961", "#ff6961"), fg_color="#3a3e41", 
                                                    image=self.checkmark_image)
                self.splice_videos_button.configure(self, text="Splice Videos", 
                                                    hover_color=("#ff6961", "#ff6961"), fg_color="#3a3e41", 
                                                    image=self.checkmark_image)
                self.quantify.configure(self, text="Class Entry",
                                        hover_color=("#ff6961", "#ff6961"), fg_color="#3a3e41", 
                                                    image=self.checkmark_image)
                self.loading_button.configure(self, text="Load Previous", 
                                                    hover_color=("#ff6961", "#ff6961"), fg_color="#3a3e41", 
                                                    image=self.checkmark_image)
            else:
                messagebox.showerror("Error", "One or more required files (labels.csv, info.txt, class.txt, image_paths.txt) not found in the folder.")
                return
        else:
            messagebox.showinfo("Information", "No folder was selected.")
            return


    def process_video(self, video_path):
        # Create a VideoCapture object
        cap = cv2.VideoCapture(video_path)

        # Check if video opened successfully
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        self.progress_bar.start()
        frame_count = 0
        while cap.isOpened():
            self.progress_bar.step()
            self.parent.update()
            ret, frame = cap.read()
            if not ret:
                break

            # Save each frame as an image
            frame_filename = f"{os.path.splitext(os.path.basename(video_path))[0]}_frame{frame_count}.jpg"
            frame_path = os.path.join(self.new_folder_path, frame_filename)
            cv2.imwrite(frame_path, frame)
            frame_count += 1
        
        self.progress_bar.stop()
        cap.release()

        print(f"Processed video: {video_path}, extracted {frame_count} frames.")

def extract_last_number(file_path):
    numbers = re.findall(r'\d+', file_path.split('/')[-1])
    return int(numbers[-1]) if numbers else 0 

def get_second_line_from_info(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if len(lines) >= 2:
            return lines[1].strip()
        else:
            return None

class ActionsWindow(CTk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        
        self.title("Reduce Frame Count")
        self.position_window(320, 195)

        self.parent = parent  # Add this line to assign the parent properly

        # Add customtkinter widgets
        label_frame_count = CTk.CTkLabel(self, text="Reduce frame Count, enter integer:")
        label_frame_count.pack(pady=10)
        
        self.entry = CTk.CTkEntry(self, width=100)
        self.entry.pack(pady=10)

        self.reduce_frame_button = CTk.CTkButton(self, text="Execute", command=self.reduce_frame_count, fg_color="#3a3e41",
                                            text_color=("gray10", "gray90"), hover_color=("#ff6961", "#ff6961"))
        self.reduce_frame_button.pack(pady=5)

        label_frame_count = CTk.CTkLabel(self, text="(will delete images and current saved data)", text_color='red')
        label_frame_count.pack(pady=10)


    def reduce_frame_count(self):  
        if not self.folder_path:
            messagebox.showerror("Error", "No data loaded.")
            return

        entered_text = self.entry.get()

        try:
            entered_number = int(entered_text)  # Attempt to convert to integer
        except ValueError:
            messagebox.showerror("Error", "Not valid integer format.")
            return 


        # List all files in the folder
        all_files = os.listdir(self.folder_path)

        # Filter to include only .jpg files
        jpg_files = [f for f in all_files if f.lower().endswith('.jpg')]

        # Create full paths for each .jpg file
        jpg_files_full_path = [os.path.join(self.folder_path, item) for item in jpg_files]
        if entered_number > len(jpg_files_full_path):
            messagebox.showerror("Error", "The number of files to keep exceeds the available .jpg files.")
            return

        # Convert to DataFrame
        jpg_files_df = pd.DataFrame(jpg_files_full_path, columns=['JPG Files'])

        # Sample n files to keep, without replacement
        jpg_files_sampled = jpg_files_df.sample(n=entered_number, replace=False)

        # Convert sampled files to a set for fast lookup
        files_to_keep = set(jpg_files_sampled['JPG Files'])

        # Loop through all .jpg files in the folder
        for file_name in jpg_files_full_path:
            # Check if the file is not in the list of files to keep
            if file_name not in files_to_keep:
                if os.path.isfile(file_name) and file_name.lower().endswith('.jpg'):  # Ensure it's a .jpg file
                    os.remove(file_name)  # Use the full path for deletion
                    print(f"Deleted: {file_name}")
                else:
                    print(f"Skipped: {file_name} (not a file or not a .jpg)")

        self.parent.load_images()
        self.parent.total_images_count = len([f for f in os.listdir(self.parent.parent.spliceFrame.new_folder_path) if f.lower().endswith('.jpg')])
        
        csv_file = os.path.join(self.folder_path, 'labels.csv')
        # Read the header (first line)
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)  # Assuming the CSV has a header

        # Write the header back, overwriting the rest of the content
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)  # Write the header back, removing the rest of the data

        self.parent.reset_info_file()
        self.parent.update_ratio_label()

    def position_window(self, width, height):
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width/2) - (width/2)
        y = (screen_height/2) - (height/2)
        self.geometry('%dx%d+%d+%d' % (width, height, x, y))

    def save_settings(self):
        print("Settings saved!")
        self.destroy()

class InfoWindow(CTk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        
        self.title("Info")
        self.position_window(630, 250)  # Increase window size to fit bigger frames

        # Create a container frame to hold the three frames side by side
        self.container_frame = CTk.CTkFrame(self)
        self.container_frame.pack(pady=10, padx=10, fill="both", expand=True)

        # Frame 1 (Adding title and label to this frame)
        self.frame1 = CTk.CTkFrame(self.container_frame, width=100, height=250)
        self.frame1.pack(side="left", padx=20)

        # Adding a title to frame1
        self.title_label1 = CTk.CTkLabel(self.frame1, text="Too many images?", font=("Arial", 16))
        self.title_label1.pack(pady=10)

        # Adding description below the title
        self.description_label1 = CTk.CTkLabel(self.frame1, 
        text=" You can reduce image count \nin the actions screen, just enter \nthe amount of frames that you \n want to score.", font=("Arial", 12))
        self.description_label1.pack(pady=10)

        # Frame 2
        self.frame2 = CTk.CTkFrame(self.container_frame, width=200, height=250)
        self.frame2.pack(side="left", padx=20)

        self.label2 = CTk.CTkLabel(self.frame2, text="Finished labeling?", font=("Arial", 16))
        self.label2.pack(pady=10)

        self.description_label2 = CTk.CTkLabel(self.frame2, 
        text="You can export the \n dataset for training down \n the road. RatSwipe offers \n a cloud compatible \n codebase for your training \n needs.", 
        font=("Arial", 12))
        self.description_label2.pack(pady=10)

        # Frame 3
        self.frame3 = CTk.CTkFrame(self.container_frame, width=200, height=250)
        self.frame3.pack(side="left", padx=20)

        self.label3 = CTk.CTkLabel(self.frame3, text="Mislabeled frame?", font=("Arial", 16))
        self.label3.pack(pady=10)

        self.description_label3 = CTk.CTkLabel(self.frame3,
        text=" You can undo your most recent \n label in the actions screen. ", font=("Arial", 12))
        self.description_label3.pack(pady=10)


    def position_window(self, width, height):
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width/2) - (width/2)
        y = (screen_height/2) - (height/2)
        self.geometry('%dx%d+%d+%d' % (width, height, x, y))

class SwipeFrame(CTk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, corner_radius=0, fg_color="transparent")
        self.parent = parent
        self.image_paths = []
        self.current_index = 0
        self.class_names = []  # This will be populated from class.txt

        self.class_colors = {
            0: "#E57373",  # Darker pastel red
            1: "#81B3D2",  # Darker pastel blue
            2: "#7DBF7D",  # Darker pastel green
            3: "#B186C4"   # Darker pastel purple
        }

        self.left_key_last_press_time = 0
        self.right_key_last_press_time = 0
        self.debounce_interval = 0.2  # 200 ms debounce interval

        # Variable to hold the last class pressed
        self.last_class = 'None'
        self.new_folder_path = None
        self.selected_folder = None

        self.create_widgets()  # Ensure the method is defined before it is called
        self.parent.bind("<KeyPress-Left>", self.key_press)
        self.parent.bind("<KeyPress-Right>", self.key_press)
        self.parent.bind("<KeyPress-a>", self.key_press)
        self.parent.bind("<KeyPress-b>", self.key_press)
        self.parent.focus_set()

    def create_widgets(self):
        # Create a frame to hold class labels
        self.class_label_frame = CTk.CTkFrame(self, fg_color="transparent")
        self.class_label_frame.pack(pady=10, padx=20, side="top", anchor="n")

        # Create a horizontal frame to hold the image and the side labels
        self.image_and_labels_frame = CTk.CTkFrame(self, fg_color="transparent")
        self.image_and_labels_frame.pack(pady=10, padx=20, fill="x", expand=True)

        # Left label to display the last pressed class
        self.left_class_label = CTk.CTkLabel(self.image_and_labels_frame, text='Last class: None', width=100)
        self.left_class_label.pack(side="left", padx=10, fill="y", expand=False)

        # Create and pack the image label for displaying images
        self.image_label = CTk.CTkLabel(self.image_and_labels_frame, text='No image loaded.', width=500, height=300)
        self.image_label.pack(side="left", pady=50)  # Adjust layout as needed

        # Right label to display the last pressed class
        self.right_class_label = CTk.CTkLabel(self.image_and_labels_frame, text='Last class: None', width=100)
        self.right_class_label.pack(side="right", padx=10, fill="y", expand=False)

        # Frame for bottom labels (this is where the controls will be restored)
        self.bottom_label_frame = CTk.CTkFrame(self, fg_color="transparent")
        self.bottom_label_frame.pack(pady=10, padx=20, side="bottom", anchor="s")

        # Ensure there are at least 4 items in the class list, filling with 'null' if necessary
        displayed_classes = self.class_names[:4] + ['null'] * (4 - len(self.class_names))

        # Add export button in the top right corner
        self.export_button = CTk.CTkButton(self, text="Export", command=self.export_data, width = 100, fg_color="#3a3e41",
                                            text_color=("gray10", "gray90"), hover_color=("#ff6961", "#ff6961"))
        self.export_button.place(x= 780, y=10, anchor="ne")  # Adjust x position for button

        # Add export button in the bottom right corner
        self.settings_button = CTk.CTkButton(self, text="Reduce Frames", command=self.open_actions_window, width = 100, fg_color="#3a3e41",
                                            text_color=("gray10", "gray90"), hover_color=("#ff6961", "#ff6961"))
        self.settings_button.place(x= 780, y=470, anchor="ne")  # Adjust x position for button

        # Add export button in the bottom right corner
        self.info_button = CTk.CTkButton(self, text="Info", command=self.open_info_window, width = 100, fg_color="#3a3e41",
                                            text_color=("gray10", "gray90"), hover_color=("#ff6961", "#ff6961"))
        self.info_button.place(x= 120, y=470, anchor="ne")  # Adjust x position for button


        # Create a label to display the ratio of scored/total slides and position it in the top left
        self.ratio_label = CTk.CTkLabel(self, text="0/0 scored", width=100)
        self.ratio_label.place(x=20, y=10)  # Place in the top left corner

        # Create the labels and buttons for controls
        mappings = [("Left D-pad", displayed_classes[0]),
                    ("Right D-pad", displayed_classes[1]),
                    ("B Button", displayed_classes[2]),
                    ("A Button", displayed_classes[3])]

        for i, (button_label, class_name) in enumerate(mappings):
            label_text = f"{button_label}: {class_name}"
            # Color the text according to the defined class color
            label = CTk.CTkLabel(self.bottom_label_frame, text=label_text, padx=10, text_color=self.class_colors.get(i, "gray"))
            label.pack(side="left", padx=10)  # Place labels horizontally with padding

    def key_press(self, event):
        """Handle key press and update the labels.csv file."""
        current_time = time.time()

        if current_time - self.left_key_last_press_time > self.debounce_interval:
            # Map keys to corresponding class names
            key_to_class_map = {
                "Left": 0,  # Class1 (fight)
                "Right": 1,  # Class2 (flee)
                "b": 2,  # Class3 (find)
                "a": 3   # Class4 (null or custom class)
            }

            class_index = key_to_class_map.get(event.keysym)

            if class_index is None or class_index >= len(self.class_names):
                return  # If no valid class is found or index is out of range, skip

            class_name = self.class_names[class_index]

            # Get the color for the pressed class
            class_color = self.class_colors.get(class_index, "#000000")  # Default to black if not found

            # Update the last class pressed and apply the relevant color
            self.last_class = class_name
            self.left_class_label.configure(text=f'Last class: {self.last_class}', text_color=class_color)
            self.right_class_label.configure(text=f'Last class: {self.last_class}', text_color=class_color)

            self.current_index = (self.current_index + 1) % len(self.image_paths)
            self.display_image()

            self.update_labels_csv(self.image_paths[self.current_index], class_name)
            self.update_info_file()
            self.update_ratio_label()

            self.left_key_last_press_time = current_time

    def update_labels_csv(self, image_path, class_name):
        """Update labels.csv with the image path and the associated class name."""
        csv_file = os.path.join(self.parent.spliceFrame.new_folder_path, 'labels.csv')
        try:
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([image_path, class_name])
        except Exception as e:
            print(f"Error updating CSV: {e}")

    def update_ratio_label(self):
        """Update the ratio label to show the number of scored images."""
        self.ratio_label.configure(text=f"{get_second_line_from_info(os.path.join(self.parent.spliceFrame.new_folder_path, 
        'info.txt'))}/{self.total_images_count} scored")

    def update_info_file(self):
        """Update the info.txt file with the current index."""
        text_path = os.path.join(self.parent.spliceFrame.new_folder_path, 'info.txt')
        try:
            with open(text_path, 'r') as file:
                lines = file.readlines()
            lines[1] = str(self.current_index)  # Update the current index in the file
            with open(text_path, 'w') as file:
                file.writelines(lines)
        except Exception as e:
            print(f"Error updating info.txt: {e}")

    def reset_info_file(self):
        """Update the info.txt file with the current index."""
        text_path = os.path.join(self.parent.spliceFrame.new_folder_path, 'info.txt')
        try:
            with open(text_path, 'r') as file:
                lines = file.readlines()
            lines[1] = '0' # Update the current index in the file
            with open(text_path, 'w') as file:
                file.writelines(lines)
        except Exception as e:
            print(f"Error updating info.txt: {e}")

    def load_images(self):
        folder_path = self.parent.spliceFrame.new_folder_path
        if folder_path and os.path.isdir(folder_path):
            self.image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]

        self.image_paths =  group_and_sort_videos(self.image_paths)
        self.total_images_count = len([f for f in os.listdir(self.parent.spliceFrame.new_folder_path) if f.lower().endswith('.jpg')])

        # Load class names from class.txt
        self.load_class_names(folder_path)

        self.display_class_labels()  # Display class labels dynamically
        self.display_bottom_labels()  # Display bottom labels dynamically

        # The output file where paths will be saved
        output_file = os.path.join(folder_path, 'image_paths.txt')

        # Open the output file in write mode
        with open(output_file, 'w') as file:
            for filename in self.image_paths:
                if filename.endswith(('.jpg', '.png')):  # Add more formats if needed
                    file.write(filename + '\n')

        # Read info.txt to get the current index (number of scored images)
        self.textPath = os.path.join(self.parent.spliceFrame.new_folder_path, 'info.txt')
        with open(self.textPath, 'r') as file:
            lines = file.readlines()
            self.current_index = int(lines[1])

        # Call update_ratio_label to refresh the label with correct values right after loading
        self.update_ratio_label()

        # Display the first image
        self.display_image()

        self.parent.statsFrame.get_csv_info()
    
    def open_actions_window(self):
        # Open the settings window as a new class
        self.actions_window = ActionsWindow(self)

    def open_info_window(self):
        # Open the settings window as a new class
        self.info_window = InfoWindow(self)

    def load_class_names(self, folder_path):
        """Load class names from class.txt and populate the class_names list."""
        class_file_path = os.path.join(folder_path, 'class.txt')
        if os.path.exists(class_file_path):
            with open(class_file_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    if line.startswith("Class list:"):
                        # Split the classes based on comma and strip whitespace
                        self.class_names = [name.strip() for name in line.split(': ')[1].split(',')]

    def display_class_labels(self):
        """Display class labels horizontally with buttons, similar to the D-pad and A/B buttons."""
        # Remove any existing labels in the frame
        for widget in self.class_label_frame.winfo_children():
            widget.destroy()  
        
        # Ensure there are at least 4 items in the class list, filling with 'null' if necessary
        displayed_classes = self.class_names[:4] + ['null'] * (4 - len(self.class_names))

        # Create mappings similar to the bottom labels, for the horizontal layout
        mappings = [("L Btn", displayed_classes[0]),
                    ("R Btn", displayed_classes[1]),
                    ("B Btn", displayed_classes[2]),
                    ("A Btn", displayed_classes[3])]

        # Display the labels and buttons
        for i, (button_label, class_name) in enumerate(mappings):
            label_text = f"{button_label}: {class_name}"
            # Apply the respective color for each class text
            label = CTk.CTkLabel(self.class_label_frame, text=label_text, padx=10, text_color=self.class_colors.get(i, "gray"))
            label.pack(side="left", padx=10)  # Place labels horizontally with padding

    def display_bottom_labels(self):
        """Display bottom labels with buttons for D-pad and A/B buttons."""
        for widget in self.bottom_label_frame.winfo_children():
            widget.destroy()  # Remove any existing labels

        # Ensure there are at least 4 items in the class list, filling with 'null' if necessary
        displayed_classes = self.class_names[:4] + ['null'] * (4 - len(self.class_names))

        # Create the labels and buttons
        mappings = [("L Btn", displayed_classes[0]),
                    ("R Btn", displayed_classes[1]),
                    ("B Btn", displayed_classes[2]),
                    ("A Btn", displayed_classes[3])]

        for i, (button_label, class_name) in enumerate(mappings):
            label_text = f"{button_label}: {class_name}"
            # Apply the respective color for each class text
            label = CTk.CTkLabel(self.bottom_label_frame, text=label_text, padx=10, text_color=self.class_colors.get(i, "gray"))
            label.pack(side="left", padx=10)  # Place labels horizontally with padding
    def display_image(self):
        """Display the current image based on self.current_index."""
        if self.image_paths:
            try:
                image_path = self.image_paths[self.current_index]
                image = Image.open(image_path)
                ctk_image = CTkImage(image, size=(500, 300))  # Convert PIL.Image to CTkImage with desired size
                self.image_label.configure(image=ctk_image, text="")  # Use CTkImage for the label, clear text
                self.image_label.image = ctk_image  # Prevent garbage collection
            except Exception as e:
                print(f"Error displaying image: {e}")
        else:
            self.image_label.configure(image=None, text="No image loaded.")
    
    def export_data(self):
        dialog = FolderNameDialog_swipe(self, title="Create Export Folder")
        folder_name = dialog.show()

        if folder_name:
            self.new_folder_path = dialog.new_folder_path  # Retrieve the newly created folder path
            self.selected_folder = self.new_folder_path  # Update the selected folder with the newly created folder path

            try:
                # Ensure that self.class_names contains the list of class names
                if not hasattr(self, 'class_names') or not self.class_names:
                    messagebox.showerror("Error", "Class names are not available.")
                    return

                class_directories = []

                for class_name in self.class_names:
                    class_directory_path = os.path.join(self.new_folder_path, class_name)
                    os.makedirs(class_directory_path, exist_ok=True)  # Create the directory if it doesn't exist
                    class_directories.append(class_directory_path)  # Store the directory path in

                messagebox.showinfo("Success", f"Folder '{folder_name}' created successfully with class directories.")
            except Exception as error:
                messagebox.showerror("Error", f"Folder or directory creation was unsuccessful: {error}")

            csv_file = os.path.join(self.parent.spliceFrame.new_folder_path, 'labels.csv')
            
            df = pd.read_csv(csv_file)
            df_unique = df.iloc[:,1].unique()
            
            for index, classifier in enumerate(df_unique):
                temp_df = df[df.iloc[:, 1] == classifier]
                file_list = temp_df['Image Path']

                destination_folder = class_directories[index]
                for _file in file_list:
                    if os.path.isfile(_file):
                        shutil.copy(_file, destination_folder)         
        else:
            messagebox.showerror("Error", "Folder creation was canceled.")

# Helper function to extract video ID and frame number
def extract_video_and_frame(filepath):
    # Extract base name (file name) from the complete file path
    filename = os.path.basename(filepath)

    # Match pattern 'vidX_frameY' from the filename
    match = re.match(r'(vid\d+)_frame(\d+)', filename)
    if match:
        video_id = match.group(1)  # Get video ID like 'vid1', 'vid2', etc.
        frame_number = int(match.group(2))  # Get frame number as an integer
        return video_id, frame_number
    return None, None

# Function to group and sort videos by frame number
def group_and_sort_videos(image_paths):
    # Dictionary to store grouped video IDs with associated frame numbers
    grouped_videos = defaultdict(list)

    # Group by video ID
    for path in image_paths:
        video_id, frame_number = extract_video_and_frame(path)
        if video_id:
            grouped_videos[video_id].append((path, frame_number))

    # Sort each group by frame number and then flatten the result back to a list of file paths
    sorted_paths = []
    for video_id in sorted(grouped_videos):  # Sort video groups by video ID
        # Sort each group by frame number
        frames_sorted = sorted(grouped_videos[video_id], key=lambda x: x[1])
        sorted_paths.extend([path for path, _ in frames_sorted])  # Append sorted paths

    return sorted_paths

class StatsFrame(CTk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, corner_radius=0, fg_color="transparent")
        self.parent = parent
        self.image_paths = []
        self.current_index = 0
        self.class_names = []  # This will be populated from class.txt

        self.create_widgets()
    
    def create_widgets(self):
        # Configure the grid layout for the main frame to have 3 equal columns
        self.grid_columnconfigure(0, weight=1, uniform="equal")
        self.grid_columnconfigure(1, weight=1, uniform="equal")
        self.grid_columnconfigure(2, weight=1, uniform="equal")
        
        # Create the three frames
        self.general_frame = CTk.CTkFrame(self, corner_radius=0)
        self.general_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.classes_frame = CTk.CTkFrame(self, corner_radius=0)
        self.classes_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        self.videos_frame = CTk.CTkFrame(self, corner_radius=0)
        self.videos_frame.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)

        # Add labels to each frame
        self.general_label = CTk.CTkLabel(self.general_frame, text="General", font=("Arial", 20, 'bold', 'underline'))
        self.general_label.pack(pady=10)

        self.classes_label = CTk.CTkLabel(self.classes_frame, text="Classes", font=("Arial", 20, 'bold','underline'))
        self.classes_label.pack(pady=10)

        self.videos_label = CTk.CTkLabel(self.videos_frame, text="Videos", font=("Arial", 20, 'bold', 'underline'))
        self.videos_label.pack(pady=10)
    
    def get_csv_info(self):
        self.folder_path = self.parent.spliceFrame.new_folder_path
        full_csv_path = os.path.join(self.folder_path, "labels.csv")
        
        # Read the CSV file
        df = pd.read_csv(full_csv_path)
        
        # Get the unique class labels and their counts
        nunique = df['Class Label'].value_counts()
        
        # Clear previous widgets from the classes_frame
        for widget in self.classes_frame.winfo_children():
            widget.destroy()
        
        # Re-add the "Classes" label at the top of the frame with underline
        self.classes_label = CTk.CTkLabel(self.classes_frame, text="Classes", font=("Arial", 20, 'bold', 'underline'))
        self.classes_label.pack(pady=10)
        
        # Add a bulleted list of class types
        bullet_header = CTk.CTkLabel(self.classes_frame, text="Class Types:", font=("Arial", 16, "bold"))
        bullet_header.pack(pady=5)
        
        for class_name in nunique.index:
            # Create a bulleted list of class types
            bullet_item = CTk.CTkLabel(self.classes_frame, text=f"• {class_name}", font=("Arial", 14))
            bullet_item.pack(pady=2)
        
        # Add a header for "Class counts:"
        header_label = CTk.CTkLabel(self.classes_frame, text="Class counts:", font=("Arial", 16, "bold"))
        header_label.pack(pady=10)
        
        # Iterate through the unique class labels and their counts
        for label, count in nunique.items():
            # Create a label for each class and count
            class_label = CTk.CTkLabel(self.classes_frame, text=f"{label}: {count}")
            class_label.pack(pady=5)
        
        return nunique

if __name__ == "__main__":
    app = Application()
    app.mainloop()