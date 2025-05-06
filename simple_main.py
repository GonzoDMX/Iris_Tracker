import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import tkinter as tk
from tkinter import ttk
import collections

# Disable PyAutoGUI failsafe (BE CAREFUL WITH THIS)
# Only disable this if you have another way to regain control!
pyautogui.FAILSAFE = False  

class ImprovedEyeTracker:
    def __init__(self):
        # Get screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Setup MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Eye indices for MediaPipe Face Mesh
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
        # Calibration points and data
        self.calibration_points = [
            (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
            (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
            (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)
        ]
        self.calibration_data = []
        self.current_point = 0
        self.mapping_matrix = None
        
        # Enhanced smoothing settings
        self.position_history = collections.deque(maxlen=15)  # Store more positions for smoother movement
        self.smoothing_factor = 0.85  # Increased smoothing factor
        self.blink_threshold = 0.2
        self.last_blink_time = 0
        self.last_mouse_pos = (self.screen_width // 2, self.screen_height // 2)
        
        # Movement constraints - keep cursor away from screen edges
        self.margin = 5  # pixels from screen edge to prevent failsafe
        
        # Flags
        self.is_calibrating = False
        self.is_tracking = False
        self.show_video = True
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Improved Eye Tracker")
        self.root.geometry("400x320")
        
        # Create frame for controls
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create buttons with better layout
        ttk.Button(control_frame, text="Start Calibration", command=self.start_calibration).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="Start Tracking", command=self.start_tracking).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="Stop Tracking", command=self.stop_tracking).pack(fill=tk.X, pady=5)
        
        # Settings frame
        settings_frame = ttk.LabelFrame(control_frame, text="Settings", padding=10)
        settings_frame.pack(fill=tk.X, pady=10)
        
        # Smoothing factor slider
        ttk.Label(settings_frame, text="Smoothing:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.smooth_var = tk.DoubleVar(value=self.smoothing_factor)
        smooth_slider = ttk.Scale(settings_frame, from_=0.5, to=0.98, 
                                  variable=self.smooth_var, orient="horizontal",
                                  command=self.update_smoothing)
        smooth_slider.grid(row=0, column=1, sticky=tk.EW, pady=5, padx=5)
        
        # Margin slider
        ttk.Label(settings_frame, text="Screen Margin:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.margin_var = tk.IntVar(value=self.margin)
        margin_slider = ttk.Scale(settings_frame, from_=1, to=50,
                                 variable=self.margin_var, orient="horizontal",
                                 command=self.update_margin)
        margin_slider.grid(row=1, column=1, sticky=tk.EW, pady=5, padx=5)
        
        # Debug checkbox
        self.debug_var = tk.BooleanVar(value=True)
        debug_check = ttk.Checkbutton(settings_frame, text="Show Debug Video", 
                                     variable=self.debug_var,
                                     command=self.toggle_debug)
        debug_check.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Status section
        status_frame = ttk.LabelFrame(control_frame, text="Status", padding=10)
        status_frame.pack(fill=tk.X, pady=5)
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.status_var, font=("Arial", 10, "bold")).pack(fill=tk.X)
        
        # Cursor position display
        self.position_var = tk.StringVar(value="Position: -")
        ttk.Label(status_frame, textvariable=self.position_var).pack(fill=tk.X)
    
    def update_smoothing(self, event=None):
        """Update smoothing factor from slider"""
        self.smoothing_factor = self.smooth_var.get()
        print(f"Smoothing factor set to {self.smoothing_factor}")
    
    def update_margin(self, event=None):
        """Update screen margin from slider"""
        self.margin = self.margin_var.get()
        print(f"Screen margin set to {self.margin}px")
    
    def toggle_debug(self):
        """Toggle debug video display"""
        self.show_video = self.debug_var.get()
        if not self.show_video and cv2.getWindowProperty('Eye Tracker', cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow('Eye Tracker')
            
    def start_calibration(self):
        """Start the calibration process"""
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        
        self.is_calibrating = True
        self.current_point = 0
        self.calibration_data = []
        
        # Create calibration window
        self.cal_window = tk.Toplevel(self.root)
        self.cal_window.attributes('-fullscreen', True)
        self.cal_window.config(bg="black")
        
        # Start calibration
        self.show_calibration_point()
    
    def show_calibration_point(self):
        """Show a calibration point on screen"""
        if not self.is_calibrating or self.current_point >= len(self.calibration_points):
            self.finish_calibration()
            return
        
        # Clear window
        for widget in self.cal_window.winfo_children():
            widget.destroy()
        
        # Calculate point position
        x_ratio, y_ratio = self.calibration_points[self.current_point]
        x = int(self.screen_width * x_ratio)
        y = int(self.screen_height * y_ratio)
        
        # Create canvas for drawing
        canvas = tk.Canvas(self.cal_window, bg="black", highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        
        # Draw target with animation (concentric circles)
        outer_circle = canvas.create_oval(x-40, y-40, x+40, y+40, outline="white", width=2)
        middle_circle = canvas.create_oval(x-20, y-20, x+20, y+20, outline="white", width=2)
        inner_circle = canvas.create_oval(x-5, y-5, x+5, y+5, fill="red", outline="red")
        
        # Draw instructions
        canvas.create_text(
            self.screen_width // 2, 50,
            text="Look at the red dot and keep your head still",
            fill="white", font=("Arial", 24)
        )
        canvas.create_text(
            self.screen_width // 2, self.screen_height - 50,
            text=f"Point {self.current_point + 1}/{len(self.calibration_points)}",
            fill="white", font=("Arial", 24)
        )
        
        # Update status
        self.status_var.set(f"Calibrating point {self.current_point + 1}/{len(self.calibration_points)}")
        
        # Animate circles to draw attention
        def animate_circles(count=0):
            if not self.is_calibrating:
                return
            
            # Pulse animation
            scale = 1.0 + 0.2 * np.sin(count * 0.2)
            canvas.itemconfig(outer_circle, outline=self.pulse_color(count))
            canvas.itemconfig(middle_circle, outline=self.pulse_color(count + 10))
            
            # Continue animation if still on this point
            if count < 10:  # animation for 1 second
                self.cal_window.after(100, lambda: animate_circles(count + 1))
            else:
                # Start collecting data after animation
                self.collect_calibration_data(x, y)
        
        # Start animation
        animate_circles()
    
    def pulse_color(self, count):
        """Return a color that pulses between white and blue"""
        intensity = int(155 + 100 * np.sin(count * 0.3))
        return f"#{intensity:02x}{intensity:02x}ff"
    
    def collect_calibration_data(self, screen_x, screen_y):
        """Collect eye data for current calibration point"""
        if not self.is_calibrating:
            return
        
        eye_positions = []
        start_time = time.time()
        duration = 2.0  # seconds per calibration point
        
        # Update status
        self.status_var.set(f"Calibrating point {self.current_point + 1} - Look at the dot")
        
        while time.time() - start_time < duration:
            # Get frame
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Process frame
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            # Get eye position if face detected
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                
                # Calculate eye centers with better landmarks
                left_eye = np.mean([[landmarks.landmark[i].x, landmarks.landmark[i].y] 
                                   for i in self.LEFT_EYE], axis=0)
                right_eye = np.mean([[landmarks.landmark[i].x, landmarks.landmark[i].y] 
                                    for i in self.RIGHT_EYE], axis=0)
                
                # Average position of both eyes
                eye_pos = np.array([(left_eye[0] + right_eye[0])/2, 
                                    (left_eye[1] + right_eye[1])/2])
                
                eye_positions.append(eye_pos)
            
            # Show frame if needed
            if self.show_video:
                if results.multi_face_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, results.multi_face_landmarks[0], self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None)
                    
                    # Draw eye centers
                    img_h, img_w = frame.shape[:2]
                    if len(eye_positions) > 0:
                        latest_pos = eye_positions[-1]
                        cv2.circle(frame, 
                                  (int(latest_pos[0] * img_w), int(latest_pos[1] * img_h)),
                                  5, (0, 255, 0), -1)
                
                # Draw progress bar
                elapsed = time.time() - start_time
                progress = int((elapsed / duration) * 100)
                cv2.rectangle(frame, (10, img_h - 40), (10 + progress * 2, img_h - 20), (0, 255, 0), -1)
                
                cv2.imshow('Calibration', frame)
                cv2.waitKey(1)
            
            # Update root to keep UI responsive
            try:
                self.root.update()
                self.cal_window.update()
            except:
                # Window might have been closed
                self.is_calibrating = False
                return
        
        # If we got data, save it
        if len(eye_positions) > 0:
            # Use median for better robustness against outliers
            median_pos = np.median(eye_positions, axis=0)
            self.calibration_data.append((median_pos, (screen_x, screen_y)))
            print(f"Calibration point {self.current_point + 1}: eye={median_pos}, screen=({screen_x}, {screen_y})")
        else:
            print(f"No data collected for point {self.current_point + 1}")
        
        # Move to next point
        self.current_point += 1
        self.show_calibration_point()
    
    def finish_calibration(self):
        """Complete calibration and calculate mapping"""
        self.is_calibrating = False
        
        # Close windows
        if hasattr(self, 'cal_window'):
            self.cal_window.destroy()
        
        if self.show_video and cv2.getWindowProperty('Calibration', cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow('Calibration')
        
        # Calculate mapping if we have enough data
        if len(self.calibration_data) >= 5:
            self.calculate_mapping()
            self.status_var.set("Calibration complete ✓")
            print("Calibration successful")
        else:
            self.status_var.set("Calibration failed - Not enough data")
            print("Calibration failed - Not enough data points")
    
    def calculate_mapping(self):
        """Calculate transformation from eye to screen coordinates"""
        eye_coords = np.array([d[0] for d in self.calibration_data])
        screen_coords = np.array([d[1] for d in self.calibration_data])
        
        # Add column of ones for affine transformation
        eye_coords_h = np.column_stack([eye_coords, np.ones(len(eye_coords))])
        
        # Solve least squares for x and y mapping separately
        x_map, _, _, _ = np.linalg.lstsq(eye_coords_h, screen_coords[:, 0], rcond=None)
        y_map, _, _, _ = np.linalg.lstsq(eye_coords_h, screen_coords[:, 1], rcond=None)
        
        self.mapping_matrix = np.vstack([x_map, y_map])
        print("Mapping matrix calculated:")
        print(self.mapping_matrix)
    
    def start_tracking(self):
        """Start eye tracking to control mouse"""
        if self.is_tracking:
            return
        
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        
        # Warn if no calibration
        if self.mapping_matrix is None:
            print("Warning: No calibration data. Using default mapping.")
            self.status_var.set("Warning: Not calibrated")
        else:
            self.status_var.set("Tracking active")
        
        # Clear position history
        self.position_history = collections.deque(maxlen=15)
        
        # Initialize position with current mouse position
        current_x, current_y = pyautogui.position()
        for _ in range(15):  # Fill history with current position
            self.position_history.append((current_x, current_y))
        
        self.is_tracking = True
        
        # Start tracking loop
        self.tracking_loop()
    
    def apply_advanced_smoothing(self, new_pos):
        """Apply advanced smoothing to create fluid motion"""
        # Add new position to history
        self.position_history.append(new_pos)
        
        # If we don't have enough history yet, just return the new position
        if len(self.position_history) < 3:
            return new_pos
        
        # Calculate weighted average with more weight on recent positions
        weights = np.linspace(0.5, 1.0, len(self.position_history))
        weights = weights / np.sum(weights)  # Normalize weights
        
        # Calculate weighted average
        x_avg = sum(pos[0] * weight for pos, weight in zip(self.position_history, weights))
        y_avg = sum(pos[1] * weight for pos, weight in zip(self.position_history, weights))
        
        # Apply additional exponential smoothing
        x_smooth = self.smoothing_factor * self.last_mouse_pos[0] + (1 - self.smoothing_factor) * x_avg
        y_smooth = self.smoothing_factor * self.last_mouse_pos[1] + (1 - self.smoothing_factor) * y_avg
        
        return (int(x_smooth), int(y_smooth))
    
    def constrain_to_screen(self, x, y):
        """Constrain coordinates to within screen bounds with margin"""
        # Apply margins to avoid triggering failsafe
        min_x = self.margin
        max_x = self.screen_width - self.margin
        min_y = self.margin
        max_y = self.screen_height - self.margin
        
        # Constrain coordinates
        x = max(min_x, min(max_x, x))
        y = max(min_y, min(max_y, y))
        
        return (x, y)
    
    def tracking_loop(self):
        """Main eye tracking loop"""
        if not self.is_tracking:
            return
        
        # Get frame
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.tracking_loop)
            return
        
        # Process frame
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            
            # Get eye positions
            left_eye = np.mean([[landmarks.landmark[i].x, landmarks.landmark[i].y] 
                               for i in self.LEFT_EYE], axis=0)
            right_eye = np.mean([[landmarks.landmark[i].x, landmarks.landmark[i].y] 
                                for i in self.RIGHT_EYE], axis=0)
            
            # Average position
            eye_pos = np.array([(left_eye[0] + right_eye[0])/2, 
                                (left_eye[1] + right_eye[1])/2])
            
            # Map to screen coordinates
            raw_screen_pos = self.map_to_screen(eye_pos)
            
            # Apply advanced smoothing
            smooth_pos = self.apply_advanced_smoothing(raw_screen_pos)
            
            # Constrain to screen bounds
            safe_pos = self.constrain_to_screen(smooth_pos[0], smooth_pos[1])
            
            # Move mouse
            pyautogui.moveTo(safe_pos[0], safe_pos[1])
            self.last_mouse_pos = safe_pos
            
            # Update position display
            self.position_var.set(f"Position: {safe_pos[0]}, {safe_pos[1]}")
            
            # Check for blink
            left_eye_open = self.calculate_eye_openness(landmarks, self.LEFT_EYE)
            right_eye_open = self.calculate_eye_openness(landmarks, self.RIGHT_EYE)
            avg_openness = (left_eye_open + right_eye_open) / 2
            
            # Show debug info on frame
            if self.show_video:
                img_h, img_w = frame.shape[:2]
                
                # Draw face mesh
                self.mp_drawing.draw_landmarks(
                    frame, landmarks, self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None)
                
                # Draw eye centers
                left_eye_px = (int(left_eye[0] * img_w), int(left_eye[1] * img_h))
                right_eye_px = (int(right_eye[0] * img_w), int(right_eye[1] * img_h))
                
                cv2.circle(frame, left_eye_px, 5, (0, 255, 0), -1)
                cv2.circle(frame, right_eye_px, 5, (0, 255, 0), -1)
                
                # Draw midpoint
                mid_px = (int((left_eye_px[0] + right_eye_px[0])/2), 
                          int((left_eye_px[1] + right_eye_px[1])/2))
                cv2.circle(frame, mid_px, 8, (0, 0, 255), -1)
                
                # Show eye openness
                cv2.putText(frame, f"Eye openness: {avg_openness:.2f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show mouse position and raw gaze position
                cv2.putText(frame, f"Mouse: {safe_pos}", (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Raw: {raw_screen_pos}", (10, 90),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show smoothing factor
                cv2.putText(frame, f"Smoothing: {self.smoothing_factor:.2f}", (10, 120),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Handle blink-to-click
            if avg_openness < self.blink_threshold:
                current_time = time.time()
                if current_time - self.last_blink_time > 1.0:
                    self.last_blink_time = current_time
                    pyautogui.click()
                    print("Blink detected - clicking")
        else:
            # No face detected
            self.position_var.set("Position: No face detected")
        
        # Display frame
        if self.show_video:
            cv2.imshow('Eye Tracker', frame)
            cv2.waitKey(1)
        
        # Continue tracking loop
        self.root.after(10, self.tracking_loop)
    
    def calculate_eye_openness(self, landmarks, eye_indices):
        """Calculate how open an eye is"""
        # Get y coordinates
        y_coords = [landmarks.landmark[i].y for i in eye_indices]
        
        # Get x coordinates for normalization
        x_coords = [landmarks.landmark[i].x for i in eye_indices]
        
        # Calculate height and width
        height = max(y_coords) - min(y_coords)
        width = max(x_coords) - min(x_coords)
        
        # Normalized height (aspect ratio)
        if width > 0:
            return height / width
        return 0
    
    def map_to_screen(self, eye_pos):
        """Map eye coordinates to screen coordinates"""
        if self.mapping_matrix is None:
            # Simple linear mapping if no calibration
            return (
                int(eye_pos[0] * self.screen_width),
                int(eye_pos[1] * self.screen_height)
            )
        else:
            # Use calibration matrix
            eye_pos_h = np.append(eye_pos, 1)  # Add 1 for affine transform
            
            # Apply mapping
            screen_x = np.dot(eye_pos_h, self.mapping_matrix[0])
            screen_y = np.dot(eye_pos_h, self.mapping_matrix[1])
            
            return (int(screen_x), int(screen_y))
    
    def stop_tracking(self):
        """Stop eye tracking"""
        self.is_tracking = False
        self.status_var.set("Tracking stopped")
        
        if self.show_video:
            if cv2.getWindowProperty('Eye Tracker', cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow('Eye Tracker')
    
    def run(self):
        """Run the application"""
        self.root.mainloop()
        
        # Clean up
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = ImprovedEyeTracker()
    tracker.run()
