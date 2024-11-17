from io import BytesIO #used to decode zip bytes
import requests #server request handling

import cv2 #frame/image display

import numpy as np #used to decode image bytes

from tkinter import filedialog  # For file selection dialogs
import easygui as eg  # For simple message dialogs

from client_to_server_verification import verify_server,verify_password #ip address and password verification functions

import os #used to access temp directory for zipfiles
import zipfile #used to open zipfiles


# Prompt the user for the server IP address
server_ip = eg.enterbox("Enter the server IP address (e.g., 127.0.0.1):", "Server Address")
if not server_ip:
    eg.msgbox("No server address provided. Exiting.", "Error")
    exit()

# Verify if the server is running
is_server_running, server_message = verify_server(server_ip)
if not is_server_running:
    eg.msgbox(f"Server verification failed: {server_message}", "Error")
    exit()

eg.msgbox(f"Server verification successful: {server_message}", "Success")

# Server URLs
url_verify_password = f"http://{server_ip}:5000/verify-password"  # password authentication
url_process_image = f"http://{server_ip}:5000/process-image"  # emotion processing

# Prompt the user for the password
password = eg.enterbox("Enter the server Password: ", "Server Password")
if not server_ip:
    eg.msgbox("No server password provided. Exiting.", "Error")
    exit()
is_verified, message = verify_password(password,url_verify_password)



if not is_verified:
    eg.msgbox(f"Authentication failed: {message}", "Error")
    exit()

eg.msgbox("Authentication successful! You can now upload an image.", "Success")

# Prompt user to select an image file
image_path = filedialog.askopenfilename(title="Select an Image File", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
if not image_path:
    eg.msgbox("No File Selected.", "Error")

# Read and process the image
image = cv2.imread(image_path)
if image is None:
    eg.msgbox("Failed to load the image file.", "Error")


# Read the image file
with open(image_path, 'rb') as img_file:
    files = {'file': img_file}

    # Send the POST request to image process url with the image
    response = requests.post(url_process_image, files=files)

    # Check if the request was successful
    if response.status_code == 200:
        # Determine the type of response based on Content-Type header
        content_type = response.headers['Content-Type']

        if content_type.startswith('image/'):
            # Single face case: Display the returned image
            np_image = np.frombuffer(response.content, np.uint8)
            processed_image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

            if processed_image is not None:
                cv2.imshow("Processed Image", processed_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("Error decoding the image.")


        elif content_type == 'application/zip':
            # Multiple faces case: Save and extract the zip file
            zip_data = BytesIO(response.content)

            with zipfile.ZipFile(zip_data, 'r') as zip_ref:
                # Extract all files to a temporary directory
                temp_dir = 'temp_faces'
                os.makedirs(temp_dir, exist_ok=True)
                zip_ref.extractall(temp_dir)
                print("Extracted frames for each detected face:")

                # Display each extracted frame
                for filename in sorted(os.listdir(temp_dir)):
                    file_path = os.path.join(temp_dir, filename)
                    extracted_image = cv2.imread(file_path)
                    if extracted_image is not None:
                        cv2.imshow(f"Processed Frame - {filename}", extracted_image)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                    else:
                        print(f"Failed to load extracted image: {filename}")

                #Clean up extracted files
                for file in os.listdir(temp_dir):
                    os.remove(os.path.join(temp_dir, file))
                os.rmdir(temp_dir)
        else:
            print("Unsupported content type:", content_type)
    else:
        print("Failed to process image:", response.json())
