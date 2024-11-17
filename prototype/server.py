#server imports
import socket

import requests
from flask import Flask, request, send_file, jsonify


import cv2 #used for image/frame display
import numpy as np #used for image decoding
from io import BytesIO #converts files to bytes

import torch  # For using PyTorch model


#Emotion Model Imports
from torchvision import transforms  # For image transformations
from model import CNNModel  # Custom emotion detection model
from data_loader import train_dataset  # Custom dataset loader

#Other utilities
from resize_image import resize_and_pad #Custom image resizer
from hash_utils import hash_data,generate_password_hash #Custom Script for password salt and hashing
from EmotionLabel import draw_text_with_border #Script for Drawing Highest Emotion Label on image

#zip file handling
import os #accesses temp directory
import tempfile #makes temporary files
import zipfile #zipfile interaction


# Set device to GPU if available, else use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained emotion detection model
num_classes = len(train_dataset.classes)  # Number of emotion classes in dataset

model = CNNModel(num_classes)

#model.load_state_dict(torch.load('models/emotion_cnn_model_20241102_235148_100epochs.pth', map_location=device))
model.load_state_dict(torch.load('models/emotion_cnn_model_20241116_084950.pth', map_location=device))

model.eval()  # Set model to evaluation mode

model = model.to(device)

# Define emotion labels from the dataset
emotion_mapping = train_dataset.classes

# Load Haar Cascade for detecting faces in images
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define image transformations for faces before model prediction
transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert to PIL image format
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((48, 48)),  # Resize to match model input size
    transforms.ToTensor()  # Convert to tensor format
])

app = Flask(__name__)


@app.route('/verify-address', methods=['POST'])
def verify_address():
    """
    Check if a server is reachable at the given IP and default port (5000).
    """
    data = request.get_json()

    # Validate input
    if not data or 'server_ip' not in data:
        return jsonify({"status": "error", "message": "Missing server_ip"}), 400

    server_ip = data['server_ip']
    port = 5000  # Default port to check

    # Attempt connection
    try:
        with socket.create_connection((server_ip, port), timeout=3):
            return jsonify({"status": "success", "message": "Server is reachable"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 502

@app.route('/verify-password', methods=['POST'])
def verify_password():
    """Verifies a password sent by the client."""
    data = request.json
    password = data.get("password")

    if not password:
        return jsonify({"error": "No password provided"}), 400

    hashed_password = hash_data(password, SERVER_SALT)
    if hashed_password == SERVER_HASH:
        return jsonify({"message": "Password verified successfully"}), 200
    else:
        return jsonify({"error": "Invalid password"}), 403


@app.route('/process-image', methods=['POST'])
def process_image():
        try:
            #Check if a file is included in the request
            if 'file' not in request.files:
                return {"error": "No file provided"}, 400

            # Read the image file from the request
            file = request.files['file']
            file_bytes = file.read()

            # Decode the image from bytes to a numpy array
            np_image = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

            if image is None: #handles non image requests
                return jsonify({"error": "Invalid image format"}), 400

            image=resize_and_pad(image) # resize image to fit frame

            # Convert image to grayscale for face detection
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) == 0:
                return jsonify({"message": "No faces detected in the image."}), 200

            if len(faces) == 1:
                # Single-face case
                x, y, w, h = faces[0]
                face_roi = gray_image[y:y + h, x:x + w]
                face_img = cv2.resize(face_roi, (48, 48))
                face_tensor = transform(face_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(face_tensor)
                    probailities = torch.softmax(output, dim=1)[0]
                    predicted_class = torch.argmax(probailities).item()
                    emotion_label = emotion_mapping[predicted_class]

                # Draw bounding box and emotion label with border text
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                draw_text_with_border(image, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Encode the annotated image as a JPEG
                _, buffer = cv2.imencode('.jpg', image)
                return send_file(BytesIO(buffer), mimetype='image/jpeg')

            # Multiple-face case
            temp_dir = tempfile.mkdtemp()
            frame_files = []

            for i, (x, y, w, h) in enumerate(faces):
                face_roi = gray_image[y:y + h, x:x + w]
                face_img = cv2.resize(face_roi, (48, 48))
                face_tensor = transform(face_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(face_tensor)
                    probabilities = torch.softmax(output, dim=1)[0]
                    predicted_class = torch.argmax(probabilities).item()
                    emotion_label = emotion_mapping[predicted_class]

                # Create a copy of the image to annotate each face separately
                face_frame = image.copy()
                cv2.rectangle(face_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                draw_text_with_border(face_frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Save each annotated frame to a temporary file
                frame_path = os.path.join(temp_dir, f'face_frame_{i}.jpg')
                cv2.imwrite(frame_path, face_frame)
                frame_files.append(frame_path)

            # Create a zip file with all annotated frames
            zip_filename = os.path.join(temp_dir, 'processed_faces.zip')
            with zipfile.ZipFile(zip_filename, 'w') as zipf:
                for frame_file in frame_files:
                    zipf.write(frame_file, os.path.basename(frame_file))

            # Return the zip file to the client
            return send_file(zip_filename, mimetype='application/zip', as_attachment=True, download_name='processed_faces.zip')

        except Exception as e:
            return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    password = input("Set a password for this session: ")
    SERVER_SALT, SERVER_HASH = generate_password_hash(password)
    app.run(host="0.0.0.0", port=5000)
