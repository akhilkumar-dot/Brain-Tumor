import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import mlflow
import mlflow.pytorch

# --- Preprocessing ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_data = datasets.ImageFolder("dataset/train", transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

val_data = datasets.ImageFolder("dataset/test", transform=transform)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=32)

# --- Model ---
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 3)  # 3 classes: cancer, tumor, aneurysm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# --- MLflow experiment ---
mlflow.set_experiment("Brain_Disease_Detection")
mlflow.start_run()
mlflow.log_param("lr", 1e-4)
mlflow.log_param("batch_size", 32)

# --- Training Loop (simplified) ---
for epoch in range(5):  # adjust epochs as needed
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = correct / total
    print(f"Epoch {epoch+1} Loss: {running_loss:.4f} Acc: {acc:.4f}")
    mlflow.log_metric("loss", running_loss)
    mlflow.log_metric("accuracy", acc)

# --- Save model ---
torch.save(model.state_dict(), "models/brain_model.pth")
mlflow.pytorch.log_model(model, "brain_model")
mlflow.end_run()
print("Training completed and model saved!")
