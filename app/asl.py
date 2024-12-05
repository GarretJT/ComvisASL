from flask import Flask, render_template, Response
import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('models/sign_language_model.h5')


# Define class labels (A-I and K-Y, excluding J and Z)
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# Flask app initialization
app = Flask(__name__)

# Initialize the webcam
camera = cv2.VideoCapture(0)

def preprocess_frame(frame):
    """
    Preprocess the frame for model prediction.
    Convert to grayscale, resize to 28x28, normalize, and reshape.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    normalized = resized.astype('float32') / 255.0
    reshaped = normalized.reshape(1, 28, 28, 1)
    return reshaped

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Preprocess the frame
            processed_frame = preprocess_frame(frame)
            
            # Predict the sign language
            predictions = model.predict(processed_frame)
            predicted_label = np.argmax(predictions, axis=1)[0]
            predicted_letter = class_labels[predicted_label]

            # Overlay prediction on the frame
            cv2.putText(frame, f"Predicted: {predicted_letter}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Encode the frame for streaming
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
