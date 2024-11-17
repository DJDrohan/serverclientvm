# transform.py
from torchvision import transforms


"""
Program Name:transform.py

Author DJ Drohan

Student Number:C21315413

Date:

Program Description: 

A program that transform dataset images into 48x48 tensors for model training/evaluation

Applies additional changes such as a horizontal flip and random rotations upto 10 degrees 
to add robustness to training

"""

# Define transformations for the dataset
# 1. Convert images to grayscale (1 color channel)
# 2. Resize to 48x48 pixels (input size for the model)
# 3. Apply random horizontal flip with 50% probability to add variety
# 4. Apply random rotation of up to 10 degrees to add robustness
# 5. Convert the image to a tensor format
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])
