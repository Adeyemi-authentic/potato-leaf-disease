
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
import io
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile

# Define CNN
class DeepPotato(nn.Module):
    def __init__(self, num_classes):
        super(DeepPotato, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Class names
CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot","Pepper__bell___healthy",
    "Potato___Early_blight","Potato___Late_blight","Potato___healthy",
    "Tomato_Bacterial_spot","Tomato_Early_blight","Tomato_Late_blight",
    "Tomato_Leaf_Mold","Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot","Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus","Tomato_healthy"
]

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = DeepPotato(num_classes=len(CLASS_NAMES)).to(device)
model.load_state_dict(torch.load("potato_cnn.pth", map_location=device))
model.eval()

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# FastAPI App
app = FastAPI()

@app.get("/")
def root():
    return {"message": "Welcome to the Potato Leaf Disease Classification API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)
            probs = F.softmax(outputs, dim=1)
            predicted_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][predicted_idx].item()

        return {
            "filename": file.filename,
            "predicted_class": CLASS_NAMES[predicted_idx],
            "confidence": round(confidence * 100, 2)
        }

    except Exception as e:
        return {"error": str(e)}
