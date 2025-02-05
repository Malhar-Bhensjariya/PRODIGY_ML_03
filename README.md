# Cat vs Dog Classification using SVM

## Overview
This project implements a **Support Vector Machine (SVM)** model to classify images of cats and dogs. The dataset consists of grayscale images resized to 64x64 pixels. The model extracts **HOG (Histogram of Oriented Gradients)** features and trains an **RBF-kernel SVM** to achieve high classification accuracy.

## Features
- **Image Preprocessing:** Converts images to grayscale and resizes them.
- **Feature Extraction:** Uses HOG features for better classification.
- **Data Normalization:** Standardizes feature values using `StandardScaler`.
- **Model Training:** Trains an SVM with optimized hyperparameters.
- **Evaluation:** Computes accuracy, confusion matrix, and classification report.

## Requirements
Ensure you have the following dependencies installed:
```bash
pip install numpy opencv-python matplotlib scikit-learn scikit-image
```

## Dataset Structure
Place your dataset in the following format:
```
/kaggle/input/task-03/train/train/
    cat.1.jpg
    cat.2.jpg
    dog.1.jpg
    dog.2.jpg
```

## Running the Project
Run the Python script:
```bash
python svm_cat_dog_classifier.py
```
This will:
1. Load and preprocess the images.
2. Extract HOG features.
3. Train the SVM model.
4. Evaluate performance with accuracy and a confusion matrix.

## Results
- Expected accuracy: **~80% or higher** (after hyperparameter tuning).
- Confusion matrix visualization to check model performance.

## Future Improvements
- Experiment with different SVM kernels.
- Use deep learning models (CNNs) for better accuracy.
- Try alternative feature extraction methods (e.g., ORB, SIFT).

## License
This project is open-source and available for educational purposes.

---
Feel free to modify and improve the model!

