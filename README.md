# Thalassemia Classifier

This is a web application that uses a deep learning model to classify blood smear images into three categories: Thalassemia Major, Thalassemia Minor, or Normal. The model is developed using PyTorch and deployed using Streamlit.

## Live Demo

https://thalassemia-classifier-cw6wcjgkbzx6nunujmkdxy.streamlit.app/

## Table of Contents

- [Overview](#overview)
- [Model Details](#model-details)
- [Installation](#installation)
- [Running the App](#running-the-app)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [How It Works](#how-it-works)
- [License](#license)

## Overview

Thalassemia is a blood disorder that requires early diagnosis for proper management. This application allows users to upload blood smear images and get predictions based on a trained convolutional neural network (CNN) model.

## Model Details

- **Architecture:** EfficientNet-B0
- **Pretrained:** ImageNet
- **Output:** Multilabel classifier with 3 outputs (major, minor, normal)
- **Loss Function:** Binary Cross Entropy
- **Optimizer:** Adam
- **Frameworks:** PyTorch, torchvision

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/thalassemia-classifier.git
   cd thalassemia-classifier
   ```

2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the trained model file `thalassemia_model.pth` is in the root directory.

## Running the App

Start the Streamlit app using:
```bash
streamlit run app.py
```

Once running, a browser window will open. You can upload an image file (jpg, jpeg, png), and the app will display predictions.

## Project Structure

```
thalassemia-classifier/
├── app.py                  # Streamlit app
├── thalassemia_model.pth   # Trained model weights
├── requirements.txt        # Python dependencies
└── README.md               # Project description
```

## Requirements

Here are the key libraries used:
- streamlit
- torch
- torchvision
- pillow
- tqdm
- pandas

See `requirements.txt` for the complete list.

## How It Works

1. Load a trained EfficientNet-B0 model with a modified output layer.
2. Upload a blood smear image via the Streamlit interface.
3. Preprocess the image (resize, normalize).
4. Make predictions using the trained model.
5. Display results for each class: Major, Minor, Normal.

## License

This project is open-source and free to use under the MIT License.

