import cv2
import mediapipe as mp
import sys

def analyze(video_path):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    
    print("Frame,Time,Thumb,Index,Middle,Ring,Pinky,WristX,WristY")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process every 5th frame to save time/output space
        if frame_count % 5 != 0:
            frame_count += 1
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        timestamp = frame_count / fps if fps > 0 else 0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = hand_landmarks.landmark
                
                # Finger States (1=Open, 0=Closed)
                # Thumb: Tip x/y vs IP x/y (using y for simplicity, assuming upright hand)
                # Actually for thumb, check distance from pinky MCP to determine if it's "out" or "in" 
                # or just simple y check if hand is vertical.
                # Let's stick to the logic in main.py for consistency.
                
                thumb_tip = landmarks[4]
                thumb_ip = landmarks[3]
                thumb_is_open = 1 if thumb_tip.y < thumb_ip.y else 0
                
                index_tip = landmarks[8]
                index_pip = landmarks[6]
                index_is_open = 1 if index_tip.y < index_pip.y else 0
                
                middle_tip = landmarks[12]
                middle_pip = landmarks[10]
                middle_is_open = 1 if middle_tip.y < middle_pip.y else 0
                
                ring_tip = landmarks[16]
                ring_pip = landmarks[14]
                ring_is_open = 1 if ring_tip.y < ring_pip.y else 0
                
                pinky_tip = landmarks[20]
                pinky_pip = landmarks[18]
                pinky_is_open = 1 if pinky_tip.y < pinky_pip.y else 0
                
                wrist_x = landmarks[0].x
                wrist_y = landmarks[0].y

                print(f"{frame_count},{timestamp:.2f},{thumb_is_open},{index_is_open},{middle_is_open},{ring_is_open},{pinky_is_open},{wrist_x:.2f},{wrist_y:.2f}")
        
        frame_count += 1

    cap.release()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze(sys.argv[1])
    else:
        print("Usage: python analyze_video.py <video_path>")
