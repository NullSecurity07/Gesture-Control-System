import os
# Force Qt to use XCB (XWayland)
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import mediapipe as mp
import face_recognition
import subprocess
import time
import config
import numpy as np
from collections import deque

class SystemControl:
    def __init__(self):
        self.last_action_time = 0

    def execute(self, command):
        current_time = time.time()
        if current_time - self.last_action_time > 1.0: 
            print(f"Executing: {command}")
            # Send desktop notification
            notification_message = config.ACTION_DESCRIPTIONS.get(command, command)
            try:
                subprocess.Popen(['notify-send', 'Gesture Control', notification_message])
            except Exception as e:
                print(f"Error sending notification: {e}")
            if command == "EXIT":
                return True
            try:
                subprocess.Popen(command, shell=True)
                self.last_action_time = current_time
            except Exception as e:
                print(f"Error executing command: {e}")
        return False

class FaceAuth:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_reference_image()
        # Add a deque to store the last few results
        self.auth_history = deque(maxlen=5)

    def load_reference_image(self):
        if os.path.exists(config.REFERENCE_IMAGE_PATH):
            try:
                image = face_recognition.load_image_file(config.REFERENCE_IMAGE_PATH)
                encoding = face_recognition.face_encodings(image)[0]
                self.known_face_encodings.append(encoding)
                self.known_face_names.append("Adithya")
                print("Reference image loaded successfully.")
            except IndexError:
                print("Error: No face found in reference image.")
            except Exception as e:
                print(f"Error loading reference image: {e}")
        else:
            print(f"Warning: {config.REFERENCE_IMAGE_PATH} not found. Authentication disabled.")

    def is_authorized(self, frame):
        if not self.known_face_encodings:
            self.auth_history.append(True)
            return True

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog") # Explicitly using hog
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        found_match = False
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=config.TOLERANCE)
            if True in matches:
                found_match = True
                break
        
        self.auth_history.append(found_match)

        # Require a majority of recent frames to be authorized
        if sum(self.auth_history) > len(self.auth_history) / 2:
            return True
        
        return False

import joblib
import pandas as pd
import numpy as np

class HandGesture:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Load Swipe Model
        try:
            self.model = joblib.load("swipe_model.pkl")
            print("Swipe model loaded.")
        except:
            print("Warning: swipe_model.pkl not found. Swipes disabled.")
            self.model = None

        # Load Static Gesture Model (SVM)
        try:
            self.static_model = joblib.load("gesture_model.pkl")
            print("Static gesture model loaded.")
        except:
            print("Warning: gesture_model.pkl not found. Static gestures disabled.")
            self.static_model = None

        # Swipe tracking (ML based)
        self.SEQ_LENGTH = 15
        self.swipe_buffer = [] # List of (x, y)
        self.last_swipe_time = 0
        self.SWIPE_COOLDOWN = 2.0
        
        # Thumbs Up Stability
        self.gesture_buffer = []
        self.BUFFER_SIZE = 20 # Increased buffer size for stability

    def get_hog_features(self, image):
        # Must match train.py
        win_size = (64, 64)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        return hog.compute(image).flatten()

    def detect(self, frame):
        # Frame is already mirrored in main loop
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        current_gesture = None
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # --- ML SWIPE DETECTION ---
                # Use Wrist(0), Thumb(4), Index(8), Pinky(20)
                wrist_x = hand_landmarks.landmark[0].x
                wrist_y = hand_landmarks.landmark[0].y
                thumb_x = hand_landmarks.landmark[4].x
                thumb_y = hand_landmarks.landmark[4].y
                index_x = hand_landmarks.landmark[8].x
                index_y = hand_landmarks.landmark[8].y
                pinky_x = hand_landmarks.landmark[20].x
                pinky_y = hand_landmarks.landmark[20].y
                
                self.swipe_buffer.append(wrist_x)
                self.swipe_buffer.append(wrist_y)
                self.swipe_buffer.append(index_x)
                self.swipe_buffer.append(index_y)
                self.swipe_buffer.append(thumb_x)
                self.swipe_buffer.append(thumb_y)
                self.swipe_buffer.append(pinky_x)
                self.swipe_buffer.append(pinky_y)
                
                # Keep only last 30 frames (240 features: 8 per frame * 30)
                if len(self.swipe_buffer) > self.SEQ_LENGTH * 8:
                    self.swipe_buffer = self.swipe_buffer[-self.SEQ_LENGTH * 8:]
                
                if self.model and len(self.swipe_buffer) == self.SEQ_LENGTH * 8 and (time.time() - self.last_swipe_time > self.SWIPE_COOLDOWN):
                    # Prepare input for model
                    cols = []
                    for i in range(self.SEQ_LENGTH):
                        cols.extend([f"wx{i}", f"wy{i}", f"ix{i}", f"iy{i}", f"tx{i}", f"ty{i}", f"px{i}", f"py{i}"])
                    
                    input_df = pd.DataFrame([self.swipe_buffer], columns=cols)
                    
                    try:
                        prediction = self.model.predict(input_df)[0]
                        
                        if prediction == "swipe_right":
                            current_gesture = "SWIPE_RIGHT"
                            self.last_swipe_time = time.time()
                            self.swipe_buffer = [] # Clear buffer
                            print("Detected: SWIPE RIGHT")
                        elif prediction == "swipe_left":
                            current_gesture = "SWIPE_LEFT"
                            self.last_swipe_time = time.time()
                            self.swipe_buffer = [] # Clear buffer
                            print("Detected: SWIPE LEFT")
                    except ValueError as e:
                        if "Feature names" in str(e):
                            print("\nCRITICAL ERROR: Model mismatch!")
                            print("You are using an old model with the new code.")
                            print("Please run 'python collect_swipes.py' then 'python train_swipes.py'\n")
                            self.model = None # Disable model to prevent spam
                        else:
                            print(f"Prediction error: {e}")
                
                # --- STATIC GESTURE DETECTION (SVM) ---
                if not current_gesture and self.static_model:
                    h, w, c = frame.shape
                    x_min, y_min = w, h
                    x_max, y_max = 0, 0
                    
                    # Calculate bounding box
                    for lm in hand_landmarks.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        if x < x_min: x_min = x
                        if x > x_max: x_max = x
                        if y < y_min: y_min = y
                        if y > y_max: y_max = y
                    
                    # Add padding
                    padding = 20
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(w, x_max + padding)
                    y_max = min(h, y_max + padding)
                    
                    # Extract ROI
                    if x_max > x_min and y_max > y_min:
                        roi = frame[y_min:y_max, x_min:x_max]
                        try:
                            # Preprocess
                            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                            resized_roi = cv2.resize(gray_roi, (64, 64))
                            
                            # Feature extraction
                            features = self.get_hog_features(resized_roi)
                            
                            # Predict
                            # 0: background, 1: thumbs_up
                            pred = self.static_model.predict([features])[0]
                            probs = self.static_model.predict_proba([features])[0]
                            
                            if pred == 1 and probs[1] > 0.85: # Increased confidence threshold
                                current_gesture = "THUMBS_UP"
                                print(f"Static Gesture: THUMBS_UP (Conf: {probs[1]:.2f})")
                                
                        except Exception as e:
                            print(f"Error in static detection: {e}")

        # Stability Check
        # Swipes are instant, pass through
        if current_gesture and "SWIPE" in current_gesture:
             return current_gesture, frame

        self.gesture_buffer.append(current_gesture)
        if len(self.gesture_buffer) > self.BUFFER_SIZE:
            self.gesture_buffer.pop(0)
            
        if len(self.gesture_buffer) == self.BUFFER_SIZE and all(g == "THUMBS_UP" for g in self.gesture_buffer):
            return "THUMBS_UP", frame
            
        return None, frame


def main():
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(3, config.FRAME_WIDTH)
    cap.set(4, config.FRAME_HEIGHT)

    auth = FaceAuth()
    gesture_detector = HandGesture()
    control = SystemControl()

    frame_count = 0
    is_user_present = False

    print("System started (MediaPipe). Show 'Thumbs Up' to EXIT.")

    cv2.namedWindow('Gesture Control', cv2.WINDOW_GUI_EXPANDED)
    try:
        cv2.setWindowProperty('Gesture Control', cv2.WND_PROP_TOPMOST, 1)
    except Exception as e:
        print(f"Warning: Could not set window to be always on top. Error: {e}")


    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1) # Mirror view
        
        # Check face every X frames
        if frame_count % config.FACE_CHECK_INTERVAL == 0:
            is_user_present = auth.is_authorized(frame)
            if is_user_present:
                print("Adithya detected.")
            else:
                print("Adithya NOT detected.")

        # Visual indicator
        color = (0, 255, 0) if is_user_present else (0, 0, 255)
        cv2.putText(frame, "AUTH" if is_user_present else "LOCKED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        gesture = None # Reset gesture at the beginning of the loop
        if is_user_present:
            gesture, frame = gesture_detector.detect(frame)
            if gesture:
                if gesture in config.ACTIONS:
                    should_exit = control.execute(config.ACTIONS[gesture])
                    if should_exit:
                        print("Exiting...")
                        break
        
        # Display gesture on frame
        if gesture:
            cv2.putText(frame, f"Gesture: {gesture}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


        cv2.imshow('Gesture Control', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()