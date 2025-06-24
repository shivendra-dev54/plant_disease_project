import torch
import os

def calculate_accuracy(y_pred, y_true):
    correct = (y_pred.argmax(dim=1) == y_true).sum().item()
    return correct / y_true.size(0)

def save_model(model, target_dir, model_name):
    os.makedirs(target_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(target_dir, model_name))
    print(f"✅ Model saved at: {target_dir}/{model_name}")

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model