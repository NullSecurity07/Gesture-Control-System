import os
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import time
import csv

# Constants
DATA_FILE = "swipe_data_cvzone.csv"
SEQ_LENGTH = 15 # Number of frames per sequence
LABELS = ["swipe_right", "swipe_left", "no_gesture"]

def collect_swipes():
    detector = HandDetector(detectionCon=0.7, maxHands=1)
    
    # Initialize CSV
    with open(DATA_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header: label, then 15 frames of (wrist, index, thumb, pinky)
        header = ["label"]
        for i in range(SEQ_LENGTH):
            header.extend([f"wx{i}", f"wy{i}", f"ix{i}", f"iy{i}", f"tx{i}", f"ty{i}", f"px{i}", f"py{i}"])
        writer.writerow(header)

    cap = cv2.VideoCapture(0)
    
    for label in LABELS:
        print(f"\n--- COLLECTING: {label.upper()} ---")
        if label == "no_gesture":
            print("Instructions: For the first 10 samples, hold the back of your hand steady. For the next 10, hold your open palm steady.")
        else:
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
                hands, frame = detector.findHands(frame, flipType=False)
                
                # Default values if hand not found
                wrist_x, wrist_y = 0, 0
                index_x, index_y = 0, 0
                thumb_x, thumb_y = 0, 0
                pinky_x, pinky_y = 0, 0
                
                if hands:
                    hand = hands[0]
                    lmList = hand["lmList"]
                    
                    # Use Wrist(0), Thumb(4), Index(8), Pinky(20)
                    wrist_x, wrist_y, _ = lmList[0]
                    thumb_x, thumb_y, _ = lmList[4]
                    index_x, index_y, _ = lmList[8]
                    pinky_x, pinky_y, _ = lmList[20]
                    
                    # Normalize coordinates
                    h, w, _ = frame.shape
                    wrist_x, wrist_y = wrist_x / w, wrist_y / h
                    thumb_x, thumb_y = thumb_x / w, thumb_y / h
                    index_x, index_y = index_x / w, index_y / h
                    pinky_x, pinky_y = pinky_x / w, pinky_y / h
                
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
