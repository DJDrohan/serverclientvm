import requests
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

# Server URL
SERVER_URL = "http://127.0.0.1:5000/process-image"  # Update with the server's IP if on a different machine

def send_image(image_path):
    # Open the image file in binary mode
    with open(image_path, 'rb') as image_file:
        # Send the image to the server
        response = requests.post(SERVER_URL, files={"file": image_file})

    # Check if the server responded successfully
    if response.status_code == 200:
        # Convert the response content into a numpy array
        np_array = np.frombuffer(response.content, np.uint8)
        processed_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        # Display the processed image
        cv2.imshow("Processed Image", processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed to process the image:", response.json())

if __name__ == "__main__":
    # Path to the image you want to send
    image_path = input("Enter the path to the image: ")
    send_image(image_path)
