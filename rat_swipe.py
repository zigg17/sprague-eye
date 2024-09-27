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
        # Update button colors in navFrame
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

        # if name == "stats":
        #     self.journalFrame.grid(row=0, column=1, sticky="nsew")
        # else:
        #     self.journalFrame.grid_forget()
        
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
        
        # Store selected folder
        self.selected_folder = selected_folder
        
        # Set window properties
        if title:
            self.title(title)
        self.geometry("300x150")  # Adjust size as needed

        # Set dark mode for the dialog if the parent is also in dark mode
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

        # Variable to store the input value
        self.folder_name = None
        self.new_folder_path = None

    def on_submit(self):
        self.folder_name = self.entry.get()
        print(self.selected_folder)  # Now this will correctly print the selected folder path
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
        
        # Set window properties
        if title:
            self.title(title)
        self.geometry("300x150")  # Adjust size as needed
        
        # Set dark mode for the dialog if the parent is also in dark mode
        CTk.set_appearance_mode("dark")
        
        # Add customtkinter widgets
        self.label = CTk.CTkLabel(self, text="Enter number of classes:")
        self.label.pack(pady=10)

        self.entry = CTk.CTkEntry(self)
        self.entry.pack(pady=10)

        self.submit_button = CTk.CTkButton(self, text="Create",
                                           fg_color=("gray75", "gray30"),  # Custom colors
                                           hover_color=("gray30", "gray75"),
                                           command=self.on_submit)
        self.submit_button.pack(pady=10)

        # Variable to store the input value
        self.folder_name = None
        self.new_folder_path = None

    def on_submit(self):
        try:
            self.class_number = int(self.entry.get())
        except ValueError:
            messagebox.showwarning("Warning", "Must be an integer.")
            return

        if self.class_number > 4:
            messagebox.showwarning("Warning", "Must be smaller than 4.")
            return

        self.destroy()

    def show(self):
        self.wait_window()
        return self.folder_name

class SpliceFrame(CTk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, corner_radius=0, fg_color="transparent")
        self.parent = parent
        self.width = 760  # Adjust the width as needed
        self.height = 100  # Adjust the height as needed
        self.configure(width=self.width, height=self.height)
        

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
        self.quantify = CTk.CTkButton(self, text="Quantify Classes", command=self.quantify_class,
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
        dialog = FolderNameDialog(self, self.selected_folder, title="Create New Folder")
        folder_name = dialog.show()
        if folder_name:
            self.desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
            self.new_folder_path = os.path.join(self.desktop_path, folder_name)
            try:
                os.makedirs(self.new_folder_path, exist_ok=True)
                self.create_folder_button.configure(self, text="Create Folder", 
                                                hover_color=("#ff6961", "#ff6961"), fg_color="#3a3e41", 
                                                image=self.checkmark_image)
            except OSError as error:
                messagebox.showerror(f"{error}", "Folder creation was canceled.")
        else:
                messagebox.showerror("Error", "Folder creation was canceled.")


    def quantify_class(self):
        dialog = ClassCountDialog(self, title= "Class Quantification")
        class_count = dialog.show()
        self.quantify.configure(self, text="Quantify Classes",
                                                hover_color=("#ff6961", "#ff6961"), fg_color="#3a3e41", 
                                                image=self.checkmark_image)
    
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
            file.write('Intialized @ ' + datetime_string + '\n0')  # Write an empty string (or you can add initial content here)
        
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
            # Check if 'labels.csv' exists and other files are '.jpg'
            if os.path.exists(os.path.join(folder_path, 'labels.csv')) and os.path.exists(os.path.join(folder_path, 'info.txt')):
                if all(file.endswith('.jpg') for file in os.listdir(folder_path) if (file != 'labels.csv' and file != 'info.txt')):
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
                    self.loading_button.configure(self, text="Load Previous", 
                                                hover_color=("#ff6961", "#ff6961"), fg_color="#3a3e41", 
                                                image=self.checkmark_image)
                    
                else:
                    messagebox.showerror("Error", "The folder does not contain only .jpg files.")
                    return
            else:
                messagebox.showerror("Error", "labels.csv or info.txt not found in the folder.")
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


class SwipeFrame(CTk.CTkFrame):
    
    def __init__(self, parent):
        super().__init__(parent, corner_radius=0, fg_color="transparent")
        self.parent = parent
        self.image_paths = []
        self.current_index = 0
    
        self.left_key_last_press_time = 0
        self.right_key_last_press_time = 0
        self.debounce_interval = 0.3  # 200 ms debounce interval
        
        self.create_widgets()
        # Bind keys to the window
        self.parent.bind("<KeyPress-Left>", self.previous_image)
        self.parent.bind("<KeyPress-Right>", self.next_image)
        self.parent.focus_set()


    def load_images(self):
        folder_path = self.parent.spliceFrame.new_folder_path
        if folder_path and os.path.isdir(folder_path):
            self.image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]

        self.image_paths = sorted(self.image_paths, key=extract_last_number)

        # The output file where paths will be saved
        output_file = os.path.join(folder_path, 'image_paths.txt')

        # Open the output file in write mode
        with open(output_file, 'w') as file:
            # Iterate over all files in the directory
            for filename in self.image_paths:
                # Check if the file is an image (e.g., .jpg or .png)
                if filename.endswith('.jpg') or filename.endswith('.png'):  # Add more formats if needed
                    # Write the path to the output file, followed by a newline
                    file.write(filename + '\n')
        
        self.textPath = os.path.join(self.parent.spliceFrame.new_folder_path, 'info.txt')
        with open(self.textPath, 'r') as file:
            lines = file.readlines()
            self.current_index = int(lines[1])
        
    def display_image(self):
        if self.image_paths:
            try:
                image_path = self.image_paths[self.current_index]
                image = Image.open(image_path)
                ctk_image = CTkImage(image, size=(500, 300))  # Convert PIL.Image to CTkImage with desired size
                self.image_label.configure(image=ctk_image)  # Use CTkImage for the label
                self.image_label.image = ctk_image  # Assign to prevent garbage collection
            except Exception as e:
                messagebox.showwarning("Warning", f"Error displaying image: {e}")

    def create_widgets(self):
        self.image_label = CTk.CTkLabel(self, text = '              ')
        self.image_label.pack(pady=100)


    def next_image(self,event):
        current_time = time.time()
        if self.image_paths and current_time - self.right_key_last_press_time > self.debounce_interval:
            # Open the CSV file in append mode and write the data
            with open(self.parent.spliceFrame.new_folder_path + '/labels.csv', 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([self.image_paths[self.current_index].split('/')[-1], 1])
            
            self.current_index = (self.current_index + 1) % len(self.image_paths)
            
            self.display_image()
            self.right_key_last_press_time = current_time
            self.textPath = os.path.join(self.parent.spliceFrame.new_folder_path, 'info.txt')

            with open(self.textPath, 'r') as file:
                lines = file.readlines()
            lines[1] = str(self.current_index)

            with open(self.textPath, 'w') as file:
                file.writelines(lines)

    def previous_image(self,event):
        current_time = time.time()
        if self.image_paths and current_time - self.left_key_last_press_time > self.debounce_interval:
            with open(self.parent.spliceFrame.new_folder_path + '/labels.csv', 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([self.image_paths[self.current_index].split('/')[-1], 0])
            self.current_index = (self.current_index + 1) % len(self.image_paths)
            self.display_image()
            self.left_key_last_press_time = current_time
            self.textPath = os.path.join(self.parent.spliceFrame.new_folder_path, 'info.txt')
            with open(self.textPath, 'r') as file:
                lines = file.readlines()
            lines[1] = str(self.current_index)
            with open(self.textPath, 'w') as file:
                file.writelines(lines)

if __name__ == "__main__":
    app = Application()
    app.mainloop()