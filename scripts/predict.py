import torch
from torchvision import transforms, models
from PIL import Image

# Load model
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load("models/brain_model.pth"))
model.eval()

# Preprocess input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
img = Image.open(r"C:\Users\Akhil M\OneDrive\Desktop\braindisease\dataset\train\aneurysm\0_2.jpg")
input_tensor = transform(img).unsqueeze(0)

# Predict
outputs = model(input_tensor)
_, predicted = outputs.max(1)
classes = ["cancer", "tumor", "aneurysm"]
print("Predicted:", classes[predicted.item()])
