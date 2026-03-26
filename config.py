import numpy as np

# Camera Config
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Face Recognition Config
REFERENCE_IMAGE_PATH = "me.jpg"
FACE_CHECK_INTERVAL = 10  # Check face every 10 frames
TOLERANCE = 0.6

# Gesture Config
# HSV Skin Color Range
LOWER_SKIN = np.array([0, 20, 70], dtype=np.uint8)
UPPER_SKIN = np.array([20, 255, 255], dtype=np.uint8)

# Minimum contour area
MIN_HAND_AREA = 15000

# Actions
ACTIONS = {
    "THUMBS_UP": "EXIT",
    "SWIPE_RIGHT": "hyprctl dispatch workspace +1",
    "SWIPE_LEFT": "hyprctl dispatch workspace -1",
}

# Cooldown between actions (frames)
ACTION_COOLDOWN = 30

ACTION_DESCRIPTIONS = {
    "EXIT": "Exiting the gesture control system.",
    "hyprctl dispatch workspace +1": "Switching to the next workspace.",
    "hyprctl dispatch workspace -1": "Switching to the previous workspace.",
}
