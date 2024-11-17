# main_modelmake.py
import torch
import os
from datetime import datetime
from train_eval import model,train_model, evaluate_model
from data_loader import train_loader, test_loader


"""
Program Name:main_modelmake.py

Author DJ Drohan

Student Number:C21315413

Date:

Program Description: 

A program that compiles an emotion detection model using components from the other model related scripts

train_eval.py
data_loader.py

transform.py (via dataloader)
model.py (via train_eval)

The max amount epochs of training are declared at first

call to model training function

call to model evaluate function

get current timestamp

name model with timestamp

save to models directory

create models directory if it doesnt exist already

make model path

save model with torch using model path and filename

"""

# Set the number of epochs for training
num_epochs = 100


# Train the model with early stopping and validation
train_model(model, train_loader, val_loader=test_loader, epochs=num_epochs, patience=5)

# Evaluate the model on the test set
evaluate_model(model, test_loader)

# Get current timestamp in a readable format
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create model filename based on timestamp and number of epochs
model_filename = f'emotion_cnn_model_{current_time}.pth'

# Define the directory to save the model
model_dir = 'models'

# Create the models directory if it doesn't exist
os.makedirs(model_dir, exist_ok=True)

# Full path to save the model
model_path = os.path.join(model_dir, model_filename)

# Save the trained model in the specified directory
torch.save(model.state_dict(), model_path)
print(f"Model saved to '{model_path}'")
