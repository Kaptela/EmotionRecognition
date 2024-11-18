import streamlit as st
import cv2
import mediapipe as mp
from torchvision import transforms, models
import torch
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 7)
    model.load_state_dict(torch.load("emotion_detection.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

emotion_classes = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Angry", "Neutral"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

st.title("Emotion Detection App")
st.write("Detect emotions in real-time via webcam or from uploaded images!")

mode = st.selectbox("Choose Mode", ["Real-Time Webcam", "Upload Image"])

if mode == "Real-Time Webcam":
    run_webcam = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])
    cap = None

    if run_webcam:
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Could not access webcam.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                    face = frame[y:y + h, x:x + w]
                    if face.size == 0:
                        continue
                    face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                    face_tensor = transform(face_pil).unsqueeze(0)

                    with torch.no_grad():
                        outputs = model(face_tensor)
                        probabilities = torch.softmax(outputs, dim=1)[0].numpy()

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, emotion_classes[np.argmax(probabilities)], (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    for i, (emotion, prob) in enumerate(zip(emotion_classes, probabilities)):
                        text = f"{emotion}: {prob * 100:.2f}%"
                        cv2.putText(frame, text, (x, y + h + 20 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            FRAME_WINDOW.image(frame, channels="BGR")

        cap.release()

elif mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0].numpy()

        st.write("Emotion Probabilities:")
        for emotion, prob in zip(emotion_classes, probabilities):
            st.write(f"{emotion}: {prob * 100:.2f}%")

        predicted_emotion = emotion_classes[np.argmax(probabilities)]
        st.write(f"Predicted Emotion: **{predicted_emotion}**")
