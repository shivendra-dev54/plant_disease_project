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

  * Directly download zip of the dataset from [PlantVillage - Kaggle](https://www.kaggle.com/emmarex/plantdisease)
  * unzip and paste 15 directories under unzipped directory (sometimes 15 dir are child of dir in unzipped dir🥲).

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

  * Update `model_path`, `image_path`, `dataset_dir` in the `inference.py` file and then run following command 
```bash
python inference.py
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
* Add visualization utilities
* Add CLI training arguments

---

## 👌 Credits

* Dataset: [PlantVillage - Kaggle](https://www.kaggle.com/emmarex/plantdisease)
* Model: ResNet-50 from PyTorch torchvision
* Assistant: Code and structure supported by [ChatGPT](https://openai.com/chatgpt)

---

## 📌 Author

**Shivendra**
Feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/shivendra-devadhe-97017a327/)!
