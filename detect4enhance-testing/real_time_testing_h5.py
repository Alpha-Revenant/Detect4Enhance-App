# Import required libraries
import cv2  # OpenCV for image capture and processing
import numpy as np  # For numerical operations
import tensorflow as tf  # For loading the trained model
import dlib  # For face and landmark detection
import matplotlib.pyplot as plt  # For displaying prediction graph
import time  # For time tracking 

# Path to the trained model (.h5 file)
MODEL_PATH_H5 = "D:/Real Time Testing/engagement_model_89.h5"

# Custom layer class to handle any custom layers during model loading
class Cast(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.identity(inputs)

# Load the trained model with custom Cast layer
model = tf.keras.models.load_model(MODEL_PATH_H5, custom_objects={"Cast": Cast}, compile=False)

# Define class labels
labels = ["Engaged", "Frustrated", "Bored", "Confused"]

# Load Dlib's face detector and facial landmark predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()

# Function to compute Eye Aspect Ratio (used for blinking/drowsiness detection)
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# Store predictions for smoothing over time
predictions_over_time = []
start_time = time.time()
graph_started = False

# Enable interactive plotting for live graph updates
plt.ion()
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, [0, 0, 0, 0], color=['#4CAF50', '#FF5722', '#03A9F4', '#9C27B0'])
ax.set_ylim(0, 100)
ax.set_ylabel("Average Percentage (%)")
ax.set_title("Live Engagement Prediction")

# Function to update the bar graph with new average predictions
def update_graph(avg_preds):
    for i, b in enumerate(bars):
        b.set_height(avg_preds[i] * 100)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Capture a frame
    if not ret or frame is None:
        print("Error: Failed to capture image")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for face detection
    faces = detector(gray)  # Detect faces in the frame

    if len(faces) > 0:
        for face in faces:
            # Get face coordinates
            x, y, x1, y1 = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)  # Draw rectangle around face

            # Crop and preprocess face for model prediction
            face_crop = frame[y:y1, x:x1]
            img = cv2.resize(face_crop, (224, 224))
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, axis=0)

            # Get facial landmarks
            shape = predictor(gray, face)
            landmarks = np.array([[p.x, p.y] for p in shape.parts()])
            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]

            # Calculate EAR and brow-eye distance for adjustment
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0
            brow_height = (landmarks[21][1] + landmarks[22][1]) / 2
            eye_height = (landmarks[37][1] + landmarks[44][1]) / 2
            frown_gap = brow_height - eye_height

            # If eyes are likely closed, assign default prediction (bored)
            if avg_ear < 0.19:
                preds = [0.146, 0.146, 0.562, 0.146]
            else:
                preds = model.predict(img)[0]  # Predict using the model

                # Adjust predictions based on facial cues
                adjustment = [0.0, 0.0, 0.0, 0.0]
                if avg_ear > 0.30 and frown_gap > -12:
                    adjustment[0] += 0.10  # More engaged
                elif frown_gap < -12:
                    adjustment[1] += 0.05  # More frustrated

                # Apply and normalize adjustments
                preds += adjustment
                preds = np.clip(preds, 0, 1)
                preds = preds / np.sum(preds)

            # Save prediction and smooth by averaging last 20 frames
            predictions_over_time.append(preds)
            if len(predictions_over_time) > 20:
                predictions_over_time.pop(0)
            avg_preds = np.mean(predictions_over_time, axis=0)

            # Update graph with averaged predictions
            update_graph(avg_preds)

            # Get final result label
            result = labels[np.argmax(avg_preds)]

            # Display prediction percentages on the frame
            for i, label in enumerate(labels):
                percent = avg_preds[i] * 100
                color = (0, 255, 0) if label == result else (255, 255, 255)
                text = f"{label}: {percent:.2f}%"
                cv2.putText(frame, text, (x, y - 10 - (i * 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Display final result label
            cv2.putText(frame, f"Result: {result}", (x, y - 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    else:
        # Show message if no face is detected
        cv2.putText(frame, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Show the video with predictions
    cv2.imshow("Real-Time Engagement Detection", frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up and close everything
plt.ioff()
plt.show()
cap.release()
cv2.destroyAllWindows()
