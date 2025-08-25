# Geothermal Temperature Prediction

A machine learning project for predicting geothermal temperatures using various regression models and deep learning techniques.

## Project Overview

This project aims to predict geothermal temperatures using geochemical features. It includes comprehensive data analysis, multiple machine learning approaches, and models evaluation.

## Project Structure

```
├── csv/                          # Dataset files
│   ├── dataset_v2.csv            # Processed dataset v2
│   ├── final_dataset.csv         # Final cleaned dataset
│
├── notebooks/
│   ├── data_cleaning_geothermal_GDR.ipynb          # Data cleaning and preprocessing
│   ├── EDA_geoTemp_prediction.ipynb                # Exploratory Data Analysis
│   ├── PCA_geoTemp_prediction.ipynb                # Principal Component Analysis
│   ├── ML_regression1_geoTemp_prediction.ipynb     # Regression models: Desicion Tree, Random Forest, XGBoost and TABPFN
│   ├── ML_regression2_geoTemp_prediction.ipynb     # Regression models: Neural networks: MLP, NN TensorFlow Keras, NN PyTorch
│   ├── ML_regression3_geoTemp_prediction.ipynb     # Regression models: Linear and GLM
│  
├── models/                       # Saved model files
└── metrics/                      # Model evaluation metrics
```

## Features

- **Data Preprocessing**: Comprehensive data cleaning and feature engineering.
- **Exploratory Data Analysis**: In-depth analysis of geochemical data patterns.
- **Multiple ML Models**: 
  - Desicion tree and Random forest
  - XGBoost and TabPFN
  - Neural networks: MLP, NN Keraas TensorFlow, NN PyTorch

- **Dimensionality Reduction**: PCA analysis
- **Model Evaluation**: Comprehensive metrics and visualization

## Models Implemented

1. **Traditional ML Models**:
   - Linear Regression
   - Random Forest Regression
   - Support Vector Regression
   - Gradient Boosting (LightGBM)

2. **Deep Learning**:
   - Multi-layer Perceptron (MLP)
   - Keras Neural Networks

3. **Dimensionality Reduction**:
   - Principal Component Analysis (PCA)

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- lightgbm
- tensorflow/keras
- jupyter

## Usage

1. Start with exploratory data analisys: `EDA_geoTemp_prediction.ipynb`
2. Run PCA analysis: `PCA_geoTemp_prediction.ipynb`
4. Experiment and train regression models: `ML_regression1/2/3_geoTemp_prediction.ipynb`

## Results

The project evaluates various models for geothermal temperature prediction, providing insights into which approaches work best for this specific domain.

## License

This project is for research and educational purposes.
