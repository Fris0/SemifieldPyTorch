import torch
import torch.nn as nn
import pandas as pd
import itertools
import math
from timeit import default_timer as timer
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from LeNet5 import SemiLeNet5, StandardLeNet5

def evaluate(model, dataloader):
    """
    Measure accuracy of model predictions

    model: LeNet5 CNN
    dataloader: Data with input and correct labels

    Output: Accuracy of model
    """ 
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    result = 100 * correct / total
    print(f"Accuracy: {result:.2f}%")
    return result

# Define column names
columns = ['kernel_size', 'semifield', 'train_time', 'accuracy']

# Create empty DataFrame with those columns
df = pd.DataFrame(columns=columns)

# Save to CSV with headers
df.to_csv('verify_and_time.csv', index=False)

# Kernel sizes to test with
kernel_sizes = [2, 3, 5, 7, 11]
# Semi-field convolutions to test with
semifields = ['MaxPlus', 'MinPlus', 'SmoothMax']

# Get product of all combinations.
kernel_padding = []
for k in kernel_sizes:
    if k == 2:
        kernel_padding.append((k, None))
    else:
        kernel_padding.append((k, 'same'))

combinations = list(itertools.product(kernel_padding, semifields))

# Crop images to 28x28
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

# Load the Fahshion MNIST data set
dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

# Split into training set and value set for testing
train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size

for ((k, pad), semifield) in combinations:
    results = []
    for i in range(30):
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Make batches of 1024 from the training set and evaluation set
        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1024)

        # Set device to cuda
        device = "cuda"

        # Create a class instance of LeNet5
        model = SemiLeNet5(num_classes=10, semi_field=semifield, kernel_size=k, padding_mode=pad).to(device)

        # Set optimizer learning rate to 0.0005
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

        # Use CrossEntropyLoss for normalized values (0 .. 1)
        criterion = nn.CrossEntropyLoss()

        # Number of iterations the training set is re-applied on the model
        num_epochs = 80

        start = timer()
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()


                running_loss += loss.item()
        end = timer()
        time = end - start
        accuracy = evaluate(model, val_loader)
        results.append({"kernel_size": k, "semifield":semifield, "train_time":time, "accuracy":accuracy})

    # Load the existing CSV
    df = pd.read_csv('verify_and_time.csv')
    df = pd.concat([df, pd.DataFrame(results)], ignore_index=True)
    df.to_csv('verify_and_time.csv', index=False)

def calculate_padding(k):
    """
    Calculate the padding for symmetric and assymetric kernels
    required for same sized outputs.
    Output: left, right top and bottom, where
    each variable represents the padding
    on that side of the input.
    """
    # Calculate total padding on height
    H = k
    p_h = H - 1
    top = left = math.floor(p_h / 2)
    bottom = right = p_h - top
    return (top, bottom)

for k in kernel_sizes:
    results = []
    for i in range(30):
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Make batches of 1024 from the training set and evaluation set
        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1024)

        # Set device to cuda
        device = "cuda"

        # Create a class instance of LeNet5
        model = None
        if k > 2:
            model = StandardLeNet5(num_classes=10, k=k, p=calculate_padding(k)).to(device)
        else:
            model = StandardLeNet5(num_classes=10, k=k, p=0).to(device)

        # Set optimizer learning rate to 0.0005
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

        # Use CrossEntropyLoss for normalized values (0 .. 1)
        criterion = nn.CrossEntropyLoss()

        # Number of iterations the training set is re-applied on the model
        num_epochs = 80

        start = timer()
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()


                running_loss += loss.item()
        end = timer()
        time = end - start
        accuracy = evaluate(model, val_loader)
        results.append({"kernel_size": k, "semifield":"Standard", "train_time":time, "accuracy":accuracy})

    # Load the existing CSV
    df = pd.read_csv('verify_and_time.csv')
    df = pd.concat([df, pd.DataFrame(results)], ignore_index=True)
    df.to_csv('verify_and_time.csv', index=False)