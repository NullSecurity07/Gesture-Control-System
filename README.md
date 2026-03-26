<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0518,50:1e1030,100:8b5cf6&height=220&section=header&text=Gesture%20Control&fontSize=64&fontColor=ffffff&fontAlignY=38&fontAlign=50&desc=Biometric%20Auth%20%26%20Kinetic%20Shell%20Automation&descSize=16&descAlignY=58&descAlign=50&descColor=c4b5fd80&animation=fadeIn" width="100%"/>

</div>

<br/>

<div align="center">

*Seamlessly map physical movement to system execution — secured by facial recognition.*

</div>

<br/>

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=rect&color=0:8b5cf6,100:1e1030&height=1&section=header" width="40%"/>
</div>

<br/>

<div align="center">

### — Overview —

</div>

<br/>

This project implements a real-time computer vision pipeline that translates hand gestures into customizable shell commands. Built with a zero-trust approach, the system remains dormant until it successfully verifies the operator's identity via facial recognition. 

Once authenticated, it processes both static (posture-based) and dynamic (motion-based) hand gestures, allowing users to execute system-level operations like switching workspaces or launching applications entirely hands-free.

<br/>

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=rect&color=0:8b5cf6,100:1e1030&height=1&section=header" width="40%"/>
</div>

<br/>

<div align="center">

### — Architecture —

</div>

<br/>

| Component | Function |
|:---|:---|
| `main.py` | Core runtime loop, camera interfacing, and execution dispatcher |
| `config.py` | Configuration state (Camera indexing, action mapping, paths) |
| `collect_*.py` | Modular scripts for capturing training data (static vs. dynamic) |
| `train_*.py` | Scikit-learn compilation scripts for generating `.pkl` models |
| `run.sh` | Shell wrapper for environment execution |

<br/>

**Execution Pipeline**

~~~text
Webcam Input (Live Feed)
    └─ Biometric Gate (face_recognition / dlib)
        ├─ Unrecognized → System Locked
        └─ Authenticated → Hand Landmark Tracking (MediaPipe)
            ├─ Static Classifier (gesture_model.pkl)
            │   └─ "Thumbs Up" → Exit Application
            └─ Dynamic Classifier (swipe_model.pkl)
                ├─ "Swipe Left" → Trigger Shell Command (e.g., Workspace Prev)
                └─ "Swipe Right" → Trigger Shell Command (e.g., Workspace Next)
~~~

<br/>

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=rect&color=0:8b5cf6,100:1e1030&height=1&section=header" width="40%"/>
</div>

<br/>

<div align="center">

### — Setup —

</div>

<br/>

**1. Environment Initialization**

~~~bash
git clone https://github.com/your-username/air-hand-gesture.git
cd air-hand-gesture

python -m venv venv_mp
source venv_mp/bin/activate  # venv_mp\Scripts\activate on Windows

pip install -r requirements.txt
~~~

**2. Biometric Configuration**

Place a clear, unobstructed image of your face in the root directory and name it `me.jpg`. The system will use this reference file to unlock the gesture controller.

**3. Action Mapping**

Edit `config.py` to map recognized gestures to your preferred OS shell commands (e.g., mapping `ACTIONS['swipe_left']` to a specific `wmctrl` or `xdotool` command).

<br/>

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=rect&color=0:8b5cf6,100:1e1030&height=1&section=header" width="40%"/>
</div>

<br/>

<div align="center">

### — Training Models —

</div>

<br/>

The system requires custom training data to accurately identify your specific hand topology and environment. 

**Static Gestures (Thumbs Up)**
~~~bash
python collect_data.py   # Follow terminal prompts to record poses
python train.py          # Compiles data into gesture_model.pkl
~~~

**Dynamic Gestures (Swipes)**
~~~bash
python collect_swipes.py # Follow terminal prompts to record motion sequences
python train_swipes.py   # Compiles data into swipe_model.pkl
~~~

<br/>

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=rect&color=0:8b5cf6,100:1e1030&height=1&section=header" width="40%"/>
</div>

<br/>

<div align="center">

### — Usage —

</div>

<br/>

Launch the main controller via the provided shell wrapper:

~~~bash
./run.sh main.py
~~~

1. Face the camera to pass the biometric check.
2. Perform configured swipe gestures to control your machine.
3. Show a "Thumbs Up" to cleanly terminate the process.

<br/>

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=rect&color=0:8b5cf6,100:1e1030&height=1&section=header" width="40%"/>
</div>

<br/>

<div align="center">

### — Stack —

</div>

<br/>

| Technology | Purpose |
|:---|:---|
| **Python** | Application logic and model training |
| **MediaPipe** | High-fidelity hand landmark detection and tracking |
| **dlib / face-recognition** | Biometric authentication and face encoding |
| **Scikit-learn** | Machine learning backend for gesture classification |
| **OpenCV** | Video capture and frame manipulation |

<br/>

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:8b5cf6,50:1e1030,100:0f0518&height=120&section=footer&animation=fadeIn" width="100%"/>

</div>
