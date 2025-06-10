import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget
)
from PyQt5.QtGui import QPixmap

# Sınıf isimlerini buraya yazıyoruz
idx_to_class = {0: "Utilities", 1: "Cavities", 2: "Intact"}
idx_to_tr = {"Utilities": "Altyapı", "Cavities": "Boşluk", "Intact": "Sağlam"}

# Model yükleme fonksiyonu
def load_model(model_path):
    num_classes = 3
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Tahmin fonksiyonu
def predict_image(model, image_path):
    img_size = 224
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        pred_class = idx_to_class[predicted.item()]
        pred_tr = idx_to_tr[pred_class]
    return pred_class, pred_tr

# PyQt5 Arayüzü
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Görsel Sınıflandırma (PyTorch)")
        self.setGeometry(100, 100, 400, 350)
        self.model = load_model("model.pth")

        self.layout = QVBoxLayout()
        self.label_img = QLabel("Henüz görsel seçilmedi")
        self.label_img.setAlignment(Qt.AlignCenter)
        self.label_pred = QLabel("Sonuç: -")
        self.label_pred.setAlignment(Qt.AlignCenter)
        self.btn = QPushButton("Görsel Seç")
        self.btn.clicked.connect(self.select_image)

        self.layout.addWidget(self.label_img)
        self.layout.addWidget(self.label_pred)
        self.layout.addWidget(self.btn)

        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Görsel Seç", "", "Resim Dosyaları (*.png *.jpg *.jpeg)")
        if file_path:
            pixmap = QPixmap(file_path).scaled(224, 224)
            self.label_img.setPixmap(pixmap)
            pred_class, pred_tr = predict_image(self.model, file_path)
            self.label_pred.setText(f"Tahmin (EN): {pred_class}\nTahmin (TR): {pred_tr}")

# PyQt5 için eksik Qt importu
from PyQt5.QtCore import Qt

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())