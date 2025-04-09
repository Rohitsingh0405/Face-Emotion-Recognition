import cv2
import os
from deepface import DeepFace

# Yaha pr Haar cascade Algorithm hai
cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
faceCascade = cv2.CascadeClassifier(cascade_path)

# Phela camera nnhi chala to dusra camera pr aana
cap = None
for camera_index in range(3):
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        print(f"Camera found at index {camera_index}")
        break
    else:
        print(f"Camera not found at index {camera_index}")
        cap.release()

if not cap or not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Yaha pr BGR use hota hai iss liye convert karo 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using Haar cascade
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    # Default emotion text
    emotion = "No face"

    try:
        # Analyze with DeepFace
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # Handle both dict and list return types
        if isinstance(result, list) and len(result) > 0:
            emotion = result[0].get('dominant_emotion', "Unknown")
        elif isinstance(result, dict):
            emotion = result.get('dominant_emotion', "Unknown")
    except Exception as e:
        print(f"DeepFace error: {e}")
        emotion = "Error"

    # For box 
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show emotion text
    cv2.putText(frame, emotion, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                (0, 0, 255), 2, cv2.LINE_AA)

    # Show frame
    cv2.imshow("Original Video", frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
