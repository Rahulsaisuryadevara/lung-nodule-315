import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# Paths to dataset
train_dir = r"C:\Users\hp\Downloads\rahultermpaper (2)\rahultermpaper\The IQ-OTHNCCD lung cancer dataset\train"
test_dir = r"C:\Users\hp\Downloads\rahultermpaper (2)\rahultermpaper\The IQ-OTHNCCD lung cancer dataset\test"

# Image transformations
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Datasets
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)


# Simple CNN Model
class LungCNN(nn.Module):
    def __init__(self):
        super(LungCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 18 * 18, 128)
        self.fc2 = nn.Linear(128, 3)  # 3 classes
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 18 * 18)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LungCNN().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
EPOCHS = 10
for epoch in range(EPOCHS):
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

    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {running_loss / len(train_loader):.4f}")

print("Training complete.")

# Save model
torch.save(model.state_dict(), "lung_cnn_model.pth")

# Save class names
with open("class_labels.txt", "w") as f:
    for idx, class_name in enumerate(train_dataset.classes):
        f.write(f"{idx}:{class_name}\n")

print("Model saved as lung_cnn_model.pth")
print("Class mapping:", train_dataset.class_to_idx)
