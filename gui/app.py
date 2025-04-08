import tkinter as tk
import cv2
from PIL import Image, ImageTk
from .display_results import display_detections

class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Initialize video source (default webcam)
        self.video_source = 0
        self.cap = cv2.VideoCapture(self.video_source)

        # Check if the video source is opened successfully
        if not self.cap.isOpened():
            print("Error: Unable to open video source")
            return

        # Get video source width and height
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create a canvas that can fit the video source size
        self.canvas = tk.Canvas(window, width=self.width, height=self.height)
        self.canvas.pack()

        # Button to take a snapshot
        self.btn_snapshot = tk.Button(window, text="Snapshot", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tk.CENTER, expand=True)

        # Dropdown to select camera
        self.camera_selector = tk.OptionMenu(window, tk.StringVar(value="Camera 0"), *self.get_available_cameras(), command=self.change_camera)
        self.camera_selector.pack(anchor=tk.CENTER, expand=True)

        # Delay for updating frames
        self.delay = 15
        self.update()

        self.window.mainloop()

    def get_available_cameras(self):
        """Detect available cameras."""
        available_cameras = []
        for i in range(5):  # Check the first 5 camera indices
            temp_cap = cv2.VideoCapture(i)
            if temp_cap.isOpened():
                available_cameras.append(f"Camera {i}")
                temp_cap.release()
        return available_cameras

    def change_camera(self, camera_label):
        """Change the video source based on user selection."""
        camera_index = int(camera_label.split(" ")[1])
        self.cap.release()  # Release the current camera
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print(f"Error: Unable to open Camera {camera_index}")
        else:
            self.video_source = camera_index

    def snapshot(self):
        """Take a snapshot of the current frame."""
        ret, frame = self.cap.read()
        if ret:
            cv2.imwrite("snapshot.png", frame)
            print("Snapshot saved as snapshot.png")

    def update(self):
        """Update the frame on the canvas."""
        ret, frame = self.cap.read()
        if ret:
            frame = display_detections(frame)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)

    def __del__(self):
        """Release the video source when the object is destroyed."""
        if self.cap.isOpened():
            self.cap.release()

App(tk.Tk(), "Tkinter and OpenCV")
