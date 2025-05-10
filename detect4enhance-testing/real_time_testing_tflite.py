import cv2
import numpy as np
import tensorflow as tf

# Load the TFLite model
MODEL_PATH_TFLITE = "D:\\Real Time Testing\\engagement_model_89.tflite"
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH_TFLITE)
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define labels
labels = ["Engaged", "Frustrated", "Bored", "Confused"]

# Load face detector (Haar Cascade) and eye detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Eye Aspect Ratio function (simplified for bounding box points)
def eye_aspect_ratio(eye_points):
    # Approximate the EAR using the horizontal and vertical distances between the bounding box points
    A = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[3]))  # Vertical distance
    B = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[2]))  # Horizontal distance
    ear = A / B  # Simplified EAR calculation
    return ear

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using OpenCV Haar Cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    result = "No face detected"  # Default result, in case no face or eye is detected

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Crop the face from the frame
            face = frame[y:y + h, x:x + w]
            img = cv2.resize(face, (224, 224))
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, axis=0)

            # Run prediction with TFLite
            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details[0]['index'])[0]
            preds = preds.astype(np.float32)  # Convert from float16 if needed

            # Detect eyes in the face region
            eyes = eye_cascade.detectMultiScale(gray[y:y + h, x:x + w])
            
            # Assume we detect at least one eye
            if len(eyes) >= 1:
                eye_points = []
                for (ex, ey, ew, eh) in eyes:
                    # Approximate the eye landmarks based on the bounding box
                    eye_center = (ex + ew // 2, ey + eh // 2)
                    eye_left = (ex, ey + eh // 2)   # Left corner of the eye
                    eye_right = (ex + ew, ey + eh // 2)  # Right corner of the eye
                    eye_top = (ex + ew // 2, ey)   # Top corner of the eye
                    eye_bottom = (ex + ew // 2, ey + eh)  # Bottom corner of the eye
                    
                    eye_points = [eye_left, eye_top, eye_right, eye_bottom]

                    # Calculate the Eye Aspect Ratio (EAR) using the approximated points
                    ear = eye_aspect_ratio(eye_points)

                    # Use EAR for detecting "Bored", "Engaged", or "Frustrated"
                    if ear < 0.20:
                        result = "Bored"
                    elif ear > 0.30:
                        result = "Engaged"
                    else:
                        result = "Confused"

                    # Display predictions
                    for i, label in enumerate(labels):
                        percent = preds[i] * 100
                        color = (0, 255, 0) if result == label and preds[i] >= 0.80 else (255, 255, 255)
                        text = f"{label}: {percent:.2f}%"
                        cv2.putText(frame, text, (x, y - 10 - (i * 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            cv2.putText(frame, f"Result: {result}", (x, y - 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    else:
        cv2.putText(frame, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Real-Time Engagement Detection (TFLite)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
