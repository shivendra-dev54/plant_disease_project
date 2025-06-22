# 🌿 Plant Disease Classification using ResNet-50

A deep learning project to classify plant diseases using the **PlantVillage** dataset and **transfer learning** with ResNet-50 in PyTorch.

---

## 📁 Project Structure

```
plant_disease_project/
├── data_setup.py          # Dataset transforms and dataloaders
├── engine.py              # Training and evaluation loops
├── inference.py           # Single image inference script
├── model.py               # ResNet-50 model creation
├── train.py               # Main training script
├── utils.py               # Accuracy, saving, early stopping
├── models/                # Saved model + class_names.json
├── data/                  # Dataset directory (from Kaggle)
└── README.md              # Project documentation
```

---

## 📦 Setup Instructions

### 1. Install Dependencies

```bash
pip install torch torchvision pillow tqdm
```

### 2. Download Dataset from Kaggle

1. Go to [https://www.kaggle.com/emmarex/plantdisease](https://www.kaggle.com/emmarex/plantdisease)
2. Download `kaggle.json` from your Kaggle account settings
3. In your project or Colab:

```python
from google.colab import files
files.upload()  # Upload kaggle.json

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d emmarex/plantdisease
!unzip -q plantdisease.zip -d data/
```

---

## 🏋️ Training the Model

Edit `train.py` if needed, then run:

```bash
python train.py
```

* Uses ResNet-50 with frozen base layers
* Random 80-20 train-validation split
* Saves model and class names to `saved_models/`

---

## 🔍 Inference on a Single Image

```bash
python inference.py \
  --model_path saved_models/resnet50_plant_disease.pth \
  --image_path path/to/image.jpg \
  --class_map saved_models/class_names.json
```

Returns predicted class and confidence.

> Make sure the number and order of classes match between training and inference.

---

## 🧠 Key Concepts

* Transfer Learning (ResNet-50 from torchvision)
* Manual train/val split with `random_split`
* Image preprocessing and normalization
* GPU support via `torch.device`
* Model checkpointing and inference

---

## 📈 Future Improvements

* Support batch inference
* Add visualization utilities (e.g., Grad-CAM)
* Add CLI training arguments
* Use Weights & Biases or TensorBoard for logging

---

## 👌 Credits

* Dataset: [PlantVillage - Kaggle](https://www.kaggle.com/emmarex/plantdisease)
* Model: ResNet-50 from PyTorch torchvision
* Assistant: Code and structure supported by [ChatGPT](https://openai.com/chatgpt)

---

## 📌 Author

**Shivendra**
Feel free to connect with me on [LinkedIn](https://www.linkedin.com/)!
