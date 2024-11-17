# data_loader.py
import os

from torch.utils.data import DataLoader
from torchvision import datasets


from transform import transform #tensor transformation script

"""
Program Name:dataloader.py

Author DJ Drohan

Student Number:C21315413

Date:

Program Description: 

A program that loads in a given dataset fro the Emotion Detection model to use for training and testing

Declares both training and testing data set using given folder names

Transforms any data within the folders into tensors using transform.py

make 64 item batches of both training and testing data

Shuffles training data 

Doesn't shuffle test data to keep evaluation consistent across runs



"""

# Define the path to the dataset directory
dataset_dir = r"shortened kaggle emotion data"

# Load the train and test datasets

#ImageFolder automatically assigns labels based on subfolder names

#Apply transformations to each dataset

train_dataset = datasets.ImageFolder(root=os.path.join(dataset_dir, 'train'), transform=transform)

test_dataset = datasets.ImageFolder(root=os.path.join(dataset_dir, 'test'), transform=transform)

# Create DataLoaders for batching and shuffling data
# 1. Shuffle the training data to improve model generalisation
# 2. No shuffling for test data to keep evaluation consistent
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
