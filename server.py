from flask import Flask, request, send_file
import cv2
import numpy as np
from io import BytesIO

app = Flask(__name__)

@app.route('/process-image', methods=['POST'])
def process_image():
    # Check if a file is included in the request
    if 'file' not in request.files:
        return {"error": "No file provided"}, 400

    # Read the image file from the request
    file = request.files['file']
    file_bytes = file.read()

    # Decode the image from bytes to a numpy array
    np_image = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    # Add "Hello" text to the image
    height, width, _ = image.shape
    cv2.putText(image, "Hello", (width // 4, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    # Encode the processed image to bytes
    _, buffer = cv2.imencode('.jpg', image)

    # Return the processed image
    return send_file(BytesIO(buffer), mimetype='image/jpeg')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
