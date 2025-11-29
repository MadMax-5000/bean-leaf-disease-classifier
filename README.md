# Bean Leaf Diagnostic Net 🌿

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-success.svg?style=for-the-badge)]()

> **Automated classification of Bean Leaf Lesions using Deep Transfer Learning.**

## 📋 Table of Contents
- [About the Project](#-about-the-project)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Performance](#-performance)
- [Roadmap](#-roadmap)
- [License](#-license)

## 🧐 About the Project

Plant diseases pose a significant threat to global food security. Early detection is key to effective disease management. **Bean Leaf Diagnostic Net** is a computer vision repository designed to classify images of bean leaves into disease categories (e.g., Angular Leaf Spot, Bean Rust, Healthy).

This project leverages **GoogleNet (Inception v1)**, using pre-trained weights from ImageNet to perform Transfer Learning. It supports two training modes:
1.  **Full Fine-Tuning:** Updating all network weights.
2.  **Feature Extraction:** Freezing the backbone and training only the classification head.

## 💾 Dataset

This project utilizes the **Bean Leaf Lesions Classification** dataset hosted on Kaggle.
- **Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/marquis03/bean-leaf-lesions-classification)
- **Input Size:** Images are resized to `128x128` (configurable).
- **Structure:** The code expects a CSV mapping file for training and validation sets.

## 🧠 Model Architecture

We utilize **GoogleNet**, a deep convolutional neural network architecture based on Inception modules. 
- **Backbone:** GoogleNet (pretrained on ImageNet).
- **Head:** A modified Fully Connected (FC) layer adapted to the specific number of disease classes in the dataset.
- **Optimizer:** Adam.
- **Loss Function:** CrossEntropyLoss.

## ⚙️ Installation

### Prerequisites
*   Python 3.8+
*   CUDA-enabled GPU (recommended)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/bean-leaf-diagnostic-net.git
   cd bean-leaf-diagnostic-net
   ```

2. Install dependencies:
   ```bash
   pip install torch torchvision matplotlib pandas numpy opendatasets scikit-learn
   ```

## 🚀 Usage

### 1. Download Data
The script uses `opendatasets` to pull data directly from Kaggle. Ensure you have your `kaggle.json` API token ready.

### 2. Training
Run the training script. By default, it will attempt to use GPU if available.

```python
# Configure your parameters in the script
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4

# Run the script
python train.py
```

### 3. Inference
```python
# Snippet for running inference on a single image
model.eval()
with torch.no_grad():
    prediction = model(input_tensor)
    predicted_class = torch.argmax(prediction, axis=1)
```

## 📊 Performance

| Experiment Mode | Epochs | Optimizer | Train Accuracy | Validation Accuracy |
|----------------|--------|-----------|----------------|---------------------|
| Full Fine-Tuning | 20     | Adam      | ~95.4% *       | ~92.1% *            |
| Feature Freeze   | 20     | Adam      | ~88.5% *       | ~86.2% *            |

*> Note: Metrics are indicative based on initial runs and may vary based on hyperparameter tuning.*

## 🗺️ Roadmap
- [x] Implement GoogleNet Backbone
- [x] Create Data Loader pipeline with augmentation
- [ ] Add ResNet50 and EfficientNet comparisons
- [ ] Export model to ONNX for mobile deployment
- [ ] Create a Streamlit Web UI for demos

## 🤝 Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License
Distributed under the MIT License. See `LICENSE` for more information.

---
*Built with ❤️ using MadMax*
