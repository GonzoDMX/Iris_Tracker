import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye indices for MediaPipe Face Mesh
# Left eye indices
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
# Right eye indices
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# For tracking eye openness
def calculate_eye_height(eye_landmarks):
    """Calculate the normalized height of the eye (for blink detection)"""
    # Find the top and bottom landmarks of the eye
    y_values = [landmark.y for landmark in eye_landmarks]
    min_y = min(y_values)
    max_y = max(y_values)
    
    # Get the eye width for normalization
    x_values = [landmark.x for landmark in eye_landmarks]
    min_x = min(x_values)
    max_x = max(x_values)
    eye_width = max_x - min_x
    
    # Normalize the eye height by width to account for distance from camera
    if eye_width > 0:
        return (max_y - min_y) / eye_width
    return 0

# Open webcam
cap = cv2.VideoCapture(0)  # Use 0 for default camera, change if needed

# Create window
cv2.namedWindow('Eye Tracking Test', cv2.WINDOW_NORMAL)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Failed to read from camera")
        break

    # Flip the image horizontally for a selfie-view display
    image = cv2.flip(image, 1)
    
    # Convert the BGR image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and find face landmarks
    results = face_mesh.process(rgb_image)
    
    # Draw the face mesh annotations on the image.
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw face mesh
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
            )
            
            # Extract the left and right eye landmarks
            left_eye_landmarks = [face_landmarks.landmark[i] for i in LEFT_EYE]
            right_eye_landmarks = [face_landmarks.landmark[i] for i in RIGHT_EYE]
            
            # Calculate eye center positions
            img_h, img_w, _ = image.shape
            left_eye_center = np.mean([[landmark.x * img_w, landmark.y * img_h] for landmark in left_eye_landmarks], axis=0).astype(int)
            right_eye_center = np.mean([[landmark.x * img_w, landmark.y * img_h] for landmark in right_eye_landmarks], axis=0).astype(int)
            
            # Draw eye centers
            cv2.circle(image, tuple(left_eye_center), 5, (0, 255, 0), -1)
            cv2.circle(image, tuple(right_eye_center), 5, (0, 255, 0), -1)
            
            # Find midpoint between eyes
            midpoint = ((left_eye_center[0] + right_eye_center[0]) // 2, 
                        (left_eye_center[1] + right_eye_center[1]) // 2)
            
            # Draw midpoint between eyes (this would be used for cursor control)
            cv2.circle(image, midpoint, 8, (0, 0, 255), -1)
            
            # Calculate eye openness
            left_eye_height = calculate_eye_height(left_eye_landmarks)
            right_eye_height = calculate_eye_height(right_eye_landmarks)
            
            # Display eye heights
            cv2.putText(image, f"Left eye: {left_eye_height:.2f}", (30, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Right eye: {right_eye_height:.2f}", (30, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Check for blink (you may need to adjust this threshold)
            blink_threshold = 0.2
            is_blinking = (left_eye_height < blink_threshold) and (right_eye_height < blink_threshold)
            
            status = "Blinking" if is_blinking else "Eyes Open"
            cv2.putText(image, status, (30, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    # Display the image
    cv2.imshow('Eye Tracking Test', image)
    
    # Exit on 'q' press
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
