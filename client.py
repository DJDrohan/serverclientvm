import requests
import cv2
import numpy as np
from io import BytesIO

# Server URL
url = "http://10.0.2.15:5000/process-image"

# Path to the image you want to send
image_path = 'input.jpg'  # Replace with your image path

# Read the image file
with open(image_path, 'rb') as img_file:
    files = {'file': img_file}
    
    # Send the POST request with the image
    response = requests.post(url, files=files)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Convert the response content to a numpy array
        np_image = np.frombuffer(response.content, np.uint8)
        # Decode the image from the numpy array
        processed_image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        
        # Display the processed image
        cv2.imshow("Processed Image", processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed to process image:", response.json())
