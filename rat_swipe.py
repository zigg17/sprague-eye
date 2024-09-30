from tkinter import filedialog
from tkinter import messagebox
import customtkinter as CTk
from customtkinter import CTkImage
from PIL import Image, ImageTk
import os
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
        self.isOpen = False
        self.swipeFrame = SwipeFrame(self)
        # Create frames
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
            self
        
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
        self.stats.configure(fg_color=("#ff5146", "#ff5146") if name == "journal" else "transparent")

    def create_widgets(self):
    # Number of columns in the grid
        # Number of columns in the grid
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
        self.geometry("300x150")

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

    def on_submit(self):
        self.folder_name = self.entry.get()
        print(self.selected_folder)
        print(self.folder_name)
        if not self.folder_name:
            messagebox.showwarning("Warning", "Folder name cannot be empty.")
            return
        
        if self.folder_name ==  os.path.basename(self.selected_folder):
            messagebox.showwarning("Warning", "Cannot be the same name as the selected folder.")
            return

        if os.path.exists(os.path.join(os.path.expanduser("~"), "Desktop", self.folder_name)):
            messagebox.showwarning("Warning", "Choose a name of a folder that doesn't exist.")
            return
        
        self.destroy()

    def show(self):
        self.wait_window()
        return self.folder_name

class ClassCountDialog(CTk.CTkToplevel): 
    def __init__(self, parent, title=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        if title:
            self.title(title)
        self.geometry("500x150") 
        
        CTk.set_appearance_mode("dark")
        
        self.label = CTk.CTkLabel(self, text="Type classes separated by comma, i.e. explore,avoid,escape")
        self.label.pack(pady=10)

        self.entry = CTk.CTkEntry(self)
        self.entry.pack(pady=10)

        self.submit_button = CTk.CTkButton(self, text="Enter",
                                           fg_color=("gray75", "gray30"),  # Custom colors
                                           hover_color=("gray30", "gray75"),
                                           command=self.on_submit)
        self.submit_button.pack(pady=10)

        self.class_string = None

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
        open(csv_file_path, 'w').close()

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

        # When everything done, release the video capture object
        cap.release()

        print(f"Processed video: {video_path}, extracted {frame_count} frames.")

def extract_last_number(file_path):
    numbers = re.findall(r'\d+', file_path.split('/')[-1])
    return int(numbers[-1]) if numbers else 0  # Use the last number, default to 0 if none

def get_second_line_from_info(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if len(lines) >= 2:
            return lines[1].strip()  # Return second line, remove any leading/trailing spaces
        else:
            return None  # Handle case if file doesn't have enough lines

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
        self.debounce_interval = 0.2  # 300 ms debounce interval

        # Variable to hold the last class pressed
        self.last_class = 'None'

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
        self.ratio_label.configure(text=f"{get_second_line_from_info(os.path.join(self.parent.spliceFrame.new_folder_path, 'info.txt'))}/{self.total_images_count} scored")

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

    def load_images(self):
        folder_path = self.parent.spliceFrame.new_folder_path
        if folder_path and os.path.isdir(folder_path):
            self.image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]

        self.image_paths = sorted(self.image_paths, key=extract_last_number)
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
        """Handle export button click."""
        # Implement your export logic here
        print("Export button clicked")


class StatsFrame(CTk.CTkFrame):
    pass

if __name__ == "__main__":
    app = Application()
    app.mainloop()