# NASA Exoplanet Hunter: Deep Learning Transit Detection System

> **Development Status:** Prototype / Research
> **Context:** NASA Space Apps Challenge / National Space Science Data Center (NSSDC) Analysis

## Project Overview
This project implements a Full Stack Deep Learning application designed to automate the detection of exoplanets using photometric data from the Kepler Space Telescope. The core component is a 1D Convolutional Neural Network (CNN) trained to identify transit signals—periodic dips in star brightness—amidst stellar flux noise.

The system serves as a functional proof-of-concept for deploying scientific machine learning models into accessible web interfaces, allowing for real-time inference on user-uploaded light curve data.

<img width="1908" height="884" alt="Screenshot 2025-12-28 at 4 48 03 PM" src="https://github.com/user-attachments/assets/bd0e28ce-7504-4901-ba90-45c245bbf4f3" />


## System Architecture
The application follows a decoupled client-server architecture:

1.  **Deep Learning Engine (Python/TensorFlow):**
    * Performs data preprocessing, signal normalization, and inference.
    * Utilizes a 5-layer CNN trained on the Kepler Exoplanet Search Results dataset.
2.  **API Layer (FastAPI):**
    * Exposes the model via REST endpoints.
    * Handles CSV file parsing, array reshaping, and serialization of results.
3.  **Visualization Interface (React.js):**
    * Renders interactive time-series plots of stellar flux.
    * Displays classification confidence metrics.

## Methodology

### 1. Data Preprocessing
The model processes the *Kepler Campaign* dataset (exoTrain/exoTest), consisting of flux intensity values over time.
* **Normalization:** StandardScaler is applied to normalize flux values to unit variance, centering the data for optimal gradient descent performance.
* **Class Imbalance Mitigation:** Synthetic Minority Over-sampling Technique (SMOTE) is utilized during training to address the scarcity of positive exoplanet examples compared to non-exoplanet noise.
* **Dimensionality:** Input data is reshaped into `(Batch_Size, Time_Steps, 1)` tensors for 1D convolution.

### 2. Neural Network Configuration
The architecture is designed to capture temporal dependencies in light curves:
* **Input Layer:** Accepts time-series flux data (3197 features).
* **Convolutional Layers:** Two 1D Conv layers (32 and 64 filters) with ReLU activation to extract local features (transit dips).
* **Pooling:** Max Pooling layers reduce dimensionality and retain the most significant signal drops.
* **Fully Connected Layer:** Dense layer (64 neurons) with Dropout (0.5) to prevent overfitting.
* **Output:** Sigmoid activation providing a binary probability score (0 = False Positive, 1 = Confirmed Exoplanet).

## Technical Stack

* **Model Training:** Python 3.9, TensorFlow, Keras, Scikit-Learn, Imbalanced-Learn (SMOTE).
* **Backend API:** FastAPI, Uvicorn, Pandas, NumPy.
* **Frontend:** React.js, Recharts (Data Visualization), Axios.


