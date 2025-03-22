
# Guava Disease Detection

## Overview

This repository provides an implementation of a Hybrid Custom CNN-based model for detecting diseases in guava plants.
The model leverages Inception-Residual blocks and incorporates Bayesian Hyperparameter Optimization to achieve high accuracy in disease classification.

## Repository Structure

- **data_preparation.py**: Script for dataset preparation, including downloading, organizing, and preprocessing images.
- **model_blocks.py**: Contains the implementation of custom Inception-Residual blocks used in the model architecture.
- **msrcnn_model.py**: Defines the complete Multi-Scale Residual Convolutional Neural Network (MSRCNN) model.
- **bayesian_optimization.py**: Performs Bayesian Hyperparameter Optimization to fine-tune model parameters.
- **train_model.py**: Script to train the MSRCNN model using the optimized hyperparameters.
- **evaluation.py**: Evaluates the trained model and generates performance metrics.
- **grad_cam.py**: Implements Grad-CAM for visualizing the regions of images that the model focuses on during prediction.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- TensorFlow
- Keras
- scikit-learn
- Matplotlib
- Seaborn
- scikit-optimize (for Bayesian Optimization)

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/imashoodnasir/Guava-Disease-Detection.git
   ```

2. **Navigate to the project directory**:

   ```bash
   cd Guava-Disease-Detection
   ```

3. **Install the required packages**:

   ```bash
   pip install -r requirements.txt
   ```

   *Note*: Ensure that the `requirements.txt` file includes all necessary dependencies.

## Usage

1. **Dataset Preparation**:

   Run `01_data_preparation.py` to download and preprocess the dataset.

   ```bash
   python 01_data_preparation.py
   ```

2. **Model Training**:

   Execute `05_train_model.py` to train the MSRCNN model.

   ```bash
   python 05_train_model.py
   ```

3. **Model Evaluation**:

   Use `06_evaluation.py` to assess the model's performance.

   ```bash
   python 06_evaluation.py
   ```

4. **Grad-CAM Visualization**:

   Run `07_grad_cam.py` to visualize the model's focus areas in the images.

   ```bash
   python 07_grad_cam.py
   ```

## Results

The model achieves high accuracy in detecting guava diseases.
Detailed performance metrics and visualizations can be found in the `results/` directory.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your improvements or bug fixes.

## License

This project is licensed under the MIT License.

---

**Note**: For detailed explanations of each script and the methodologies used, please refer to the comments within the code files.
