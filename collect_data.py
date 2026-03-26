import os
# Force Qt to use XCB (XWayland) to avoid Wayland plugin errors
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import time

DATA_DIR = "data"
LABELS = ["thumbs_up", "background"]
IMG_SIZE = (64, 64) 

def create_dirs():
    for label in LABELS:
        path = os.path.join(DATA_DIR, label)
        if not os.path.exists(path):
            os.makedirs(path)

def collect_automated(label, num_samples=100):
    cap = cv2.VideoCapture(0)
    print(f"\n--- Collecting data for: {label} ---")
    print(f"I will take {num_samples} photos automatically.")
    print("Get ready! Capture starts in 3 seconds.")
    
    # Initial countdown
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    print("GO!")

    count = len(os.listdir(os.path.join(DATA_DIR, label)))
    captured = 0
    
    while captured < num_samples:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        
        # ROI Box
        cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)
        
        # Display info
        cv2.putText(frame, f"Collecting: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Captured: {captured}/{num_samples}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow(f"Collecting {label}", frame)
        cv2.waitKey(1)
        
        # Save frame
        roi = frame[100:400, 100:400]
        roi = cv2.resize(roi, IMG_SIZE)
        filename = os.path.join(DATA_DIR, label, f"{count}.jpg")
        cv2.imwrite(filename, roi)
        
        count += 1
        captured += 1
        
        # Small delay to allow movement
        time.sleep(0.1) 
            
    cap.release()
    cv2.destroyAllWindows()
    print(f"Finished collecting {label}.")

if __name__ == "__main__":
    create_dirs()
    print("Welcome to the Automated Data Collector.")
    
    # Thumbs Up Collection
    print("\nSTEP 1: Thumbs Up")
    print("Instructions: Keep your hand inside the green box. Rotate it slightly, move it a bit closer/further, and change angles while the script captures.")
    input("Press Enter to start capturing 'thumbs_up'...")
    collect_automated("thumbs_up", num_samples=150)
    
    # Background Collection
    print("\nSTEP 2: Background / Negative")
    print("Instructions: Remove your hand from the box. You can also put your hand in and do OTHER gestures (open palm, fist, etc.) that are NOT thumbs up.")
    input("Press Enter to start capturing 'background'...")
    collect_automated("background", num_samples=150)
        
    print("\nData collection complete. Let me know when you are done.")
