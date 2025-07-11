import torch
from data_setup import create_dataloaders
from model import create_model
from engine import train_one_epoch, eval_model
from utils import save_model
import torch.multiprocessing

def main():
    # Config
    data_dir = './data'
    batch_size = 32
    num_epochs = 1
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, val_loader, num_classes = create_dataloaders(data_dir=data_dir, val_split=0.2, batch_size=batch_size)

    # Load model
    model = create_model(num_classes=num_classes, freeze_features=True).to(device)

    # Loss & optimizer 
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")

        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = eval_model(model, val_loader, loss_fn, device)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")

    # Save final model 
    save_model(model, "models", "resnet50_plant_disease.pth")

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
