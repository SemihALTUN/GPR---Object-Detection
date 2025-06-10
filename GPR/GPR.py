import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import h5py #HDF5

def save_torch_model_h5(model, filename="model.h5"):
    weights = model.state_dict()
    with h5py.File(filename, 'w') as f:
        for k, v in weights.items():
            f.create_dataset(k, data=v.cpu().numpy())

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(epochs):
        # Eğitim
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc.item())

        # Doğrulama
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)

        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = val_running_corrects.double() / len(val_loader.dataset)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc.item())

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

    return train_loss_history, val_loss_history, train_acc_history, val_acc_history

def main():
    # GPU kontrolü
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Parametreler
    batch_size = 32
    epochs = 20
    num_classes = 3
    img_size = 224

    # Klasör yolları
    data_dir = "Data"

    # Transformlar
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # Dataset ve Dataloader
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform)

    # Windows için multiprocessing sorunu yaşamamak adına num_workers=0 da kullanabilirsin!
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print("Class indices:", train_dataset.class_to_idx)

    # Model
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # Loss ve optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Eğitim
    train_loss, val_loss, train_acc, val_acc = train_model(
        model, train_loader, val_loader, criterion, optimizer, epochs, device
    )

    # Modeli kaydet
    torch.save(model.state_dict(), "model.pth")
    save_torch_model_h5(model, "model.h5")

    # Eğitim grafikleri
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.title("Loss")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(train_acc, label="Train Acc")
    plt.plot(val_acc, label="Val Acc")
    plt.title("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_plots.png")
    plt.show()

    # Test doğruluğu
    evaluate(model, test_loader, device)

if __name__ == "__main__":
    main()
