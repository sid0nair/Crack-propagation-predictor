# Crack Path Prediction Using U-Net in PyTorch

This project implements a machine learning model to predict the final crack propagation path in a pre-notched glass plate under tensile loading. The model utilizes a U-Net–style convolutional neural network built with PyTorch and is designed to work on full-color images. An 80/20 train–test split is applied on a dataset of 49 image pairs. The project is intended to be run on Google Colab with GPU acceleration (T4).

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Environment Setup](#environment-setup)
- [Usage](#usage)
  - [Running in Google Colab](#running-in-google-colab)
  - [Running Locally](#running-locally)
- [Training](#training)
- [Evaluation](#evaluation)
## Project Overview

This project builds a deep learning solution to predict the crack propagation path in a glass plate. Starting with an initial configuration image, the network predicts the final configuration (i.e., the crack path) under a fixed tensile load. The main components include:

- **Data Loading & Preprocessing:** Custom PyTorch dataset to handle full-color images.
- **Model Architecture:** A U-Net–style network that processes 3-channel images.
- **Training & Testing Pipeline:** An 80/20 data split with training on 39 samples and evaluation on 10 samples.
- **Visualization:** Output comparisons between the input, ground truth, and predicted final configurations.

## Dataset

- **Structure:**  
  The dataset must be organized into two folders: /content/data/ input/ # Full-color images of the initial plate configuration. output/ # Full-color images of the final plate configuration after crack propagation.

- **Size:**  
The dataset includes 49 image pairs. An 80% (≈39 images) training set and 20% (≈10 images) test set split is performed within the code.

## Model Architecture

The implemented U-Net network consists of the following:
- **Encoder:** Several convolutional layers with ReLU activations and max-pooling to extract spatial features.
- **Decoder:** Transpose convolution layers for upsampling combined with skip connections from the encoder.
- **Output:** A final convolution layer with a sigmoid activation function to produce normalized 3-channel output images.

## Environment Setup

To run this project on Google Colab or locally, make sure that the following dependencies are installed:

- Python 3.11.12
- [PyTorch](https://pytorch.org)
- [torchvision](https://pytorch.org/vision/stable/index.html)
- [OpenCV](https://opencv.org) (for image processing)
- [Matplotlib](https://matplotlib.org) (for visualization)

### Installation Example (in a Colab cell or terminal):

```bash
!pip install torch torchvision opencv-python matplotlib
