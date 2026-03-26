import os
# Force Qt to use XCB (XWayland) to avoid Wayland plugin errors
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import mediapipe as mp
import numpy as np
import time
import csv

# Constants
# Constants
DATA_FILE = "swipe_data.csv"
SEQ_LENGTH = 15 # Number of frames per sequence
LABELS = ["swipe_right", "swipe_left", "no_gesture"]

def collect_swipes():
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Initialize CSV
    should_write_header = True
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            header_line = f.readline()
            if "tx0" not in header_line: # Check for new schema (Thumb X)
                print(f"Old data format detected. Backing up to {DATA_FILE}.bak")
                os.rename(DATA_FILE, f"{DATA_FILE}.bak")
                should_write_header = True
            else:
                should_write_header = False
    
    if should_write_header:
        with open(DATA_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header: label, then 30 frames of (wrist, index, thumb, pinky)
            header = ["label"]
            for i in range(SEQ_LENGTH):
                header.extend([f"wx{i}", f"wy{i}", f"ix{i}", f"iy{i}", f"tx{i}", f"ty{i}", f"px{i}", f"py{i}"])
            writer.writerow(header)

    cap = cv2.VideoCapture(0)
    
    for label in LABELS:
        print(f"\n--- COLLECTING: {label.upper()} ---")
        print("Instructions: Perform the gesture naturally when you see 'GO'.")
        print("We will collect 20 samples.")
        input("Press Enter to start...")
        
        for i in range(20):
            print(f"Sample {i+1}/20 - Get Ready...")
            time.sleep(1)
            print("GO!")
            
            sequence = []
            for _ in range(SEQ_LENGTH):
                ret, frame = cap.read()
                if not ret: break
                
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                
                # Default values if hand not found
                wrist_x, wrist_y = 0, 0
                index_x, index_y = 0, 0
                thumb_x, thumb_y = 0, 0
                pinky_x, pinky_y = 0, 0
                
                if results.multi_hand_landmarks:
                    # Use Wrist(0), Thumb(4), Index(8), Pinky(20)
                    lm_wrist = results.multi_hand_landmarks[0].landmark[0]
                    lm_thumb = results.multi_hand_landmarks[0].landmark[4]
                    lm_index = results.multi_hand_landmarks[0].landmark[8]
                    lm_pinky = results.multi_hand_landmarks[0].landmark[20]
                    
                    wrist_x, wrist_y = lm_wrist.x, lm_wrist.y
                    thumb_x, thumb_y = lm_thumb.x, lm_thumb.y
                    index_x, index_y = lm_index.x, lm_index.y
                    pinky_x, pinky_y = lm_pinky.x, lm_pinky.y
                    
                    mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
                
                sequence.extend([wrist_x, wrist_y, index_x, index_y, thumb_x, thumb_y, pinky_x, pinky_y])
                
                cv2.putText(frame, f"Recording {label} {i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Collector", frame)
                cv2.waitKey(1)
            
            # Save sequence
            with open(DATA_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([label] + sequence)
                
            print("Saved.")
            time.sleep(0.5)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Data saved to {DATA_FILE}")

if __name__ == "__main__":
    collect_swipes()
