# Emotion Recognition App

This project is a machine learning-based application that detects emotions from images and real-time webcam input using a fine-tuned ResNet18 model.

---

## Project Link

Access the GitHub repository for this project:
[Emotion Recognition Repository](https://github.com/Kaptela/EmotionRecognition.git)

---

## Features

- **Real-time Emotion Detection**: Captures and analyzes emotions directly from a webcam feed.
- **Image Upload**: Detects emotions in uploaded images.
- **Emotion Classes**: Identifies emotions such as `Surprise`, `Fear`, `Disgust`, `Happy`, `Sad`, `Angry`, and `Neutral`.

---

## Instructions for Running the Code

### Clone the Repository
Clone the project to a local machine:
```bash
git clone https://github.com/Kaptela/EmotionRecognition.git
cd EmotionRecognition
```

### Install Dependencies
Ensure Python (>=3.8) is installed. Install required libraries using:
```bash
pip install -r requirements.txt
```

### Run the Application
Start the Streamlit app:
```bash
streamlit run app.py
```

### Using the App
- **Real-Time Webcam Mode**: Select the "Real-Time Webcam" option to detect emotions in live video.
- **Upload Image Mode**: Upload an image to classify emotions in one or more faces.

### Pre-trained Model Weights
Ensure the file `emotion_detection.pth` (containing the trained ResNet18 weights) is located in the same directory as `app.py`.

## Hardware
- **Webcam**: Required for real-time detection.
- **GPU (optional)**: Recommended for faster inference but not mandatory.

## Technologies Used
- **Python**: The main programming language for the project.
- **PyTorch**: For model training and inference.
- **Streamlit**: For building the interactive web app.
- **Mediapipe**: For real-time face detection.
- **OpenCV**: For processing webcam video frames.
- **Pillow**: For image preprocessing.

## Notes
- This app was created for educational purposes.
- Contributions, issues, and pull requests are welcome to improve the project.

## Contact
If you have any questions or suggestions, feel free to reach out:
- **GitHub**: [Kaptela](https://github.com/Kaptela)



