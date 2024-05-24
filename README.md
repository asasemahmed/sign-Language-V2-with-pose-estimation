# Sign Language Translator using MediaPipe 🚀

**Description:**

This project implements a real-time sign language translation system using MediaPipe's hand pose estimation and a deep learning model. It aims to bridge the communication gap between sign language users and those who don't understand it.

**Features:**

- Leverages MediaPipe for robust hand pose detection and landmark extraction.
- Integrates a machine learning model random forest for sign language recognition.
- Provides real-time translation of recognized signs into text.

**Getting Started:**

1. **Prerequisites:** 🏅
   - Python 3.x (check with `python --version`)
   - Required libraries:
     - `mediapipe` (`pip install mediapipe`)
     - sklearn for machine learning models
   - Install additional dependencies as specified in your model's documentation (if using a pre-trained model).

2. **Clone the Repository:**
   ```bash
   git clone https://github.com/asasemahmed/sign-Language-V2-with-pose-estimation.git
   ```

3. **Run the Translator:** 📢🎙️
   ```bash
   python main.py
   ```

   This will launch a webcam stream where you can sign in front of the camera. Recognized signs will be translated and displayed in real-time.

**Contributing:**

We welcome contributions to this project! Please feel free to:

- Create pull requests for bug fixes, improvements, or new features.
- Raise issues for any problems you encounter.
- Open discussions on potential enhancements.
