import cv2
import customtkinter as ctk
from PIL import Image, ImageTk

class VideoPlayer:
    def __init__(self, root, video_source):
        self.root = root
        self.root.title("CustomTkinter Video Player")
        self.root.geometry("800x600")
        ctk.set_appearance_mode("System")  # Modes: "System" (default), "Dark", "Light"
        ctk.set_default_color_theme("blue")  # Themes: "blue" (default), "green", "dark-blue"

        # Initialize video capture
        self.cap = cv2.VideoCapture(video_source)
        
        # Fixed video frame size
        self.video_width = 640
        self.video_height = 360

        # Control state
        self.paused = False

        # Main layout frame
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Scrollable frame to the left
        self.scrollable_frame = ctk.CTkScrollableFrame(self.main_frame, width=150)
        self.scrollable_frame.pack(side="left", fill="y", padx=10)

        # Add items to the scrollable frame (for demonstration purposes)
        for i in range(20):
            button = ctk.CTkButton(self.scrollable_frame, text=f"Item {i+1}")
            button.pack(pady=5, padx=5)

        # Video display frame to the right
        self.video_frame = ctk.CTkFrame(self.main_frame, width=self.video_width, height=self.video_height)
        self.video_frame.pack(side="left", padx=10, pady=10)

        # Label for displaying video
        self.label = ctk.CTkLabel(self.video_frame, text="")
        self.label.pack()

        # Play/Pause button
        self.play_pause_button = ctk.CTkButton(
            self.root, text="Pause", command=self.toggle_pause
        )
        self.play_pause_button.pack(pady=10)

        # Bind spacebar for pause/resume
        self.root.bind("<space>", lambda event: self.toggle_pause())

        # Start video playback
        self.play_video()
    
    def play_video(self):
        if not self.paused:
            ret, frame = self.cap.read()
            if ret:
                # Resize the frame to fit the fixed video frame size
                frame = cv2.resize(frame, (self.video_width, self.video_height))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.label.imgtk = imgtk
                self.label.configure(image=imgtk)
        
        # Schedule next frame
        self.label.after(10, self.play_video)
    
    def toggle_pause(self):
        self.paused = not self.paused
        self.play_pause_button.configure(text="Play" if self.paused else "Pause")
    
    def on_close(self):
        self.cap.release()
        self.root.destroy()

# Initialize CustomTkinter App
root = ctk.CTk()
player = VideoPlayer(root, 'video.mp4')

# Handle window close
root.protocol("WM_DELETE_WINDOW", player.on_close)
root.mainloop()