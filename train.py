import pandas as pd
import numpy as np
import yaml
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.pytorch


# get the folder where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# set mlflow to store data in the same folder as the script
mlflow.set_tracking_uri(f"file:///{os.path.join(SCRIPT_DIR, 'mlruns')}")


# ============================================================
# Step 1: Load hyperparameters from config file
# ============================================================
config_path = os.path.join(SCRIPT_DIR, "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

learning_rate = config["learning_rate"]
epochs = config["epochs"]
batch_size = config["batch_size"]

print("Hyperparameters loaded from config.yaml:")
print(f"  learning_rate = {learning_rate}")
print(f"  epochs        = {epochs}")
print(f"  batch_size    = {batch_size}")


# ============================================================
# Step 2: Load CSV data
# ============================================================
csv_path = os.path.join(SCRIPT_DIR, "fashion-mnist_train.csv")
print("\nLoading data from fashion-mnist_train.csv ...")
df = pd.read_csv(csv_path)

# separate labels and pixel values
labels = df["label"].values
pixels = df.drop(columns=["label"]).values

# normalize pixel values to 0-1
pixels = pixels / 255.0

# convert to pytorch tensors
X_tensor = torch.tensor(pixels, dtype=torch.float32)
y_tensor = torch.tensor(labels, dtype=torch.long)

# split into train and test (80% train, 20% test)
split = int(0.8 * len(X_tensor))
X_train, X_test = X_tensor[:split], X_tensor[split:]
y_train, y_test = y_tensor[:split], y_tensor[split:]

# create dataloaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Training samples: {len(X_train)}")
print(f"Test samples:     {len(X_test)}")


# ============================================================
# Step 3: Define a simple neural network model
# ============================================================
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


model = SimpleNet()
print("\nModel created:")
print(model)


# ============================================================
# Step 4: Set up loss function and optimizer
# ============================================================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# ============================================================
# Step 5: MLflow tracking
# ============================================================
# set experiment name
mlflow.set_experiment("Assignment3_Ahmed")

# start mlflow run
with mlflow.start_run():

    # log parameters
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)

    # set tag
    mlflow.set_tag("student_id", "Ahmed_123")

    # ========================================================
    # Step 6: Training loop
    # ========================================================
    print("\n--- Training Started ---")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            # forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # track stats
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        # calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total

        # log metrics to mlflow (live logging every epoch)
        mlflow.log_metric("train_loss", epoch_loss, step=epoch)
        mlflow.log_metric("train_accuracy", epoch_accuracy, step=epoch)

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} - Accuracy: {epoch_accuracy:.4f}")

    print("--- Training Finished ---\n")

    # ========================================================
    # Step 7: Evaluate on test set
    # ========================================================
    model.eval()
    test_correct = 0
    test_total = 0
    test_loss = 0.0

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            test_total += batch_y.size(0)
            test_correct += (predicted == batch_y).sum().item()

    final_test_loss = test_loss / len(test_loader)
    final_test_accuracy = test_correct / test_total

    # log final test metrics
    mlflow.log_metric("test_loss", final_test_loss)
    mlflow.log_metric("test_accuracy", final_test_accuracy)

    print(f"Test Loss:     {final_test_loss:.4f}")
    print(f"Test Accuracy: {final_test_accuracy:.4f}")

    # ========================================================
    # Step 8: Save model using MLflow Model Flavor
    # ========================================================
    mlflow.pytorch.log_model(model, "fashion_mnist_model")
    print("\nModel saved to MLflow artifacts.")

print("\nDone! Check MLflow UI at http://localhost:5000")
