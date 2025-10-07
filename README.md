# 🧠 Brain MRI Disease Classification using Deep Learning

An advanced deep learning project that classifies brain CT scans into multiple disease categories — **Tumor**, **Cancer**, **Aneurysm**, or **Normal**.  
This model leverages Convolutional Neural Networks (CNNs) with modern preprocessing, augmentation, and interpretability techniques to aid in early neurological diagnosis.

---

## 🎯 Project Overview

The goal of this project is to build a **multi-class image classification** system capable of analyzing brain CT scans and automatically predicting the type of disease present.  
By automating the classification process, this model can assist doctors, researchers, and radiologists in faster and more reliable diagnostics.

---

## 🧩 Features

- 🧠 **Multi-disease prediction**: Detects Tumor, Cancer, Aneurysm, or Normal CT scans.  
- 🧮 **Deep learning pipeline**: Data preprocessing → Augmentation → Model training → Evaluation.  
- 🧑‍⚕️ **Clinically oriented** output with class labels and confidence scores.  
- 🧰 **Modular scripts** for preprocessing, training, and inference.  
- 📈 **Performance tracking** with MLflow.  
- 🧾 **Dataset versioning** managed via DVC for reproducibility.  

---

## 📂 Repository Structure

Brain-disease/
├── dataset.dvc
├── mlruns/ # MLflow experiment tracking
├── scripts/
│ ├── preprocess.py # Data cleaning, resizing, and augmentation
│ ├── train.py # CNN model training & evaluation
│ └── inference.py # Run predictions on new CT images
├── models/ # Saved trained models
├── main.py # Entry point to run the entire workflow
├── requirements.txt
└── README.md


---

## 🧠 Model Architecture

- **Model Type:** Custom Convolutional Neural Network (CNN)  
- **Layers Used:** Conv2D, MaxPooling, Dropout, Dense (Softmax output)  
- **Activation:** ReLU (hidden), Softmax (output)  
- **Loss Function:** Categorical Cross-Entropy  
- **Optimizer:** Adam  
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix  

---

## 🧰 Tech Stack

| Layer | Tool / Library |
|-------|----------------|
| Deep Learning | TensorFlow / Keras |
| Data Handling | NumPy, Pandas, OpenCV, PIL |
| Visualization | Matplotlib, Seaborn |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| Environment | Python 3.10+ |

---

## 📦 Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/akhilkumar-dot/Brain-Tumor.git
cd Brain-Tumor
