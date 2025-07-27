# 🥦 Vegetable Image Classifier using CNN

A Convolutional Neural Network (CNN)-based image classification model that can identify different types of vegetables from images. Built using TensorFlow and trained on a labeled image dataset of vegetables.

---

## 🧠 Project Overview

This project implements an image classification pipeline using **deep learning (CNN)** to classify vegetable images into predefined categories. It leverages TensorFlow/Keras tools for data loading, preprocessing, training, and evaluation.

---

## 🔍 Features

- ✅ Built with **Convolutional Neural Networks**
- ✅ Uses **image_dataset_from_directory** for efficient loading
- ✅ Includes training-validation accuracy visualization
- ✅ Predicts custom images with label visualization
- ✅ Model saved as `.h5` file for reuse and deployment

---

## 🖼️ Dataset

- Source: [Kaggle Vegetable Image Dataset](https://www.kaggle.com/datasets/kristianmik/vegetable-image-dataset)
- Classes include: Broccoli, Tomato, Carrot, etc.
- Image Size: Resized to 180x180
- Organized by folders: one folder per class

---

## 🏗️ Model Architecture

```python
Sequential([
    Rescaling(1./255),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(),
    Conv2D(16, (3,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])
