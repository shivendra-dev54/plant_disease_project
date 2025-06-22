from pathlib import Path
import torch
from torchvision import transforms, models, datasets
from torch import nn
import matplotlib.pyplot as plt
from PIL import Image
import os

def create_model(num_classes, freeze_features=True):
    model = models.resnet50(pretrained=False)  # pretrained=False since we load weights
    if freeze_features:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def load_class_names_from_dataset(dataset_path):
    dataset = datasets.ImageFolder(dataset_path)
    return dataset.classes

def predict_image(model, image_path, class_names, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_class = torch.max(probs, 1)

    predicted_label = class_names[pred_class.item()]
    return predicted_label, confidence.item()

def main(model_path, image_path, dataset_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names = load_class_names_from_dataset(dataset_dir)
    num_classes = len(class_names)

    model = create_model(num_classes=num_classes, freeze_features=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    label, confidence = predict_image(model, image_path, class_names, device)
    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Prediction: {label} ({confidence * 100:.2f}%)")
    plt.show()
    print(f"\n🔍 Prediction: {label} ({confidence * 100:.2f}% confidence)")

if __name__ == "__main__":
    model_path = Path("./models/resnet50_plant_disease.pth")
    image_path = Path("./data/Tomato_Septoria_leaf_spot/0a25f893-1b5f-4845-baa1-f68ac03d96ac___Matt.S_CG 7863.JPG")
    dataset_dir = Path("./data")
    
    main(model_path, image_path, dataset_dir)
