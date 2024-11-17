#train_eval.py
import copy

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from data_loader import train_dataset
from model import CNNModel, init_weights


"""
Program Name:train_eval.py

Author DJ Drohan

Student Number:C21315413

Date:

Program Description: 

A program that Trains and evaluates A model using A Convolutional neural network
present within model.py

This program Trains the model with the loaded Dataset (Kaggle Facial Emotion Dataset)

Runs it for a given amount of Epochs Sent in by main_modelmake.py

And Evaluates the accuracy of the model by comparing it to the testing dataset

This program uses A cuda based GPU if available for faster and more accurate training

The program also has a patience feature if a drop in accuracy from the previous epoch occurs a 
given amount of times the training stops and the model is saved to prevent further accuracy loss


"""

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model, loss function, optimizer, and learning rate scheduler
num_classes = len(train_dataset.classes)
model = CNNModel(num_classes).to(device)
model.apply(init_weights)  # Apply custom weight initialization

criterion = nn.CrossEntropyLoss()  # Loss function for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with learning rate 0.001
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # Decays LR every 5 epochs by 0.5


# Training function with early stopping to prevent overfitting
def train_model(model, train_loader, val_loader=None, epochs=10, patience=5):
    best_accuracy = 0  # Track the best accuracy achieved
    best_model_weights = copy.deepcopy(model.state_dict())  # Store best model weights
    no_improvement = 0  # Track epochs without improvement

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0

        # Training loop for each batch
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  # Reset gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            running_loss += loss.item()

        scheduler.step()  # Adjust learning rate after each epoch

        # Optional validation step with early stopping
        if val_loader:
            val_accuracy = evaluate_model(model, val_loader)
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model_weights = copy.deepcopy(model.state_dict())
                no_improvement = 0
                #ive noticed that with large epoch runs that 1 epoch might have a
                #performance dip that is rectified afterward


            else:
                no_improvement += 1

            if no_improvement >= patience:  # Stop if no improvement for `patience` epochs
                print("Early stopping...")
                break

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}, {device}')

    # Load best weights
    model.load_state_dict(best_model_weights)


# Evaluation function to calculate accuracy on the test set
def evaluate_model(model, test_loader):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    # Disable gradient calculation for faster evaluation
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # Get model outputs
            _, predicted = torch.max(outputs.data, 1)  # Predicted class is the one with the highest score
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total  # Calculate accuracy
    print(f'Accuracy: {accuracy:.2f}%')
    return accuracy
