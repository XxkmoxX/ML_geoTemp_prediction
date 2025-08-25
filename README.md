# Geothermal Temperature Prediction

A machine learning project for predicting geothermal temperatures using various regression models and deep learning techniques.

## Project Overview

This project aims to predict geothermal temperatures using geochemical data. It includes comprehensive data analysis, multiple machine learning approaches, and model evaluation.

## Project Structure

```
├── csv/                          # Dataset files
│   ├── dataset_GDR_raw.csv       # Raw dataset
│   ├── dataset_v2.csv            # Processed dataset v2
│   ├── final_dataset.csv         # Final cleaned dataset
│   └── back/                     # Backup dataset versions
├── notebooks/
│   ├── data_cleaning_geothermal_GDR.ipynb    # Data cleaning and preprocessing
│   ├── EDA_geoTemp_prediction.ipynb          # Exploratory Data Analysis
│   ├── PCA_geoTemp_prediction.ipynb          # Principal Component Analysis
│   ├── ML_regression1_geoTemp_prediction.ipynb  # Regression models - Part 1
│   ├── ML_regression2_geoTemp_prediction.ipynb  # Regression models - Part 2
│   ├── ML_regression3_geoTemp_prediction.ipynb  # Regression models - Part 3
│   └── ml_nn-keras_temp_prediction.ipynb     # Neural Network models
├── lightgbm-regression-project/   # LightGBM implementation
├── models/                       # Saved model files
└── metrics/                      # Model evaluation metrics
```

## Features

- **Data Preprocessing**: Comprehensive data cleaning and feature engineering
- **Exploratory Data Analysis**: In-depth analysis of geochemical data patterns
- **Multiple ML Models**: 
  - Linear Regression
  - Random Forest
  - Support Vector Regression
  - LightGBM
  - Neural Networks (Keras)
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

1. Start with data cleaning: `data_cleaning_geothermal_GDR.ipynb`
2. Explore the data: `EDA_geoTemp_prediction.ipynb`
3. Run PCA analysis: `PCA_geoTemp_prediction.ipynb`
4. Train regression models: `ML_regression1/2/3_geoTemp_prediction.ipynb`
5. Experiment with neural networks: `ml_nn-keras_temp_prediction.ipynb`

## Results

The project evaluates various models for geothermal temperature prediction, providing insights into which approaches work best for this specific domain.

## License

This project is for research and educational purposes.
