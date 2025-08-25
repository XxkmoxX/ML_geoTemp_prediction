# Geothermal Temperature Prediction

A comprehensive machine learning project for predicting geothermal temperatures using various regression models, deep learning techniques, and advanced ensemble methods.

## Project Overview

This project aims to predict geothermal temperatures using geochemical features through a systematic approach that includes data preprocessing, exploratory data analysis, dimensionality reduction, and evaluation of multiple machine learning models. The project provides a thorough comparison of traditional ML algorithms and modern deep learning approaches.

## Project Structure

```
geoTemp_prediction/
├── csv/                                          # Dataset files
│   ├── dataset_v2.csv                            # Processed dataset v2
│   └── final_dataset.csv                         # Final cleaned dataset
│
├── notebooks/
│   ├── EDA_geoTemp_prediction.ipynb              # Exploratory Data Analysis
│   ├── PCA_geoTemp_prediction.ipynb              # Principal Component Analysis
│   ├── ML_regression1_geoTemp_prediction.ipynb   # Tree-based models: Decision Tree, Random Forest, XGBoost and advanced model: TabPFN
│   ├── ML_regression2_geoTemp_prediction.ipynb   # Neural Networks: MLP, NN TensorFlow Keras
│   ├── metrics_comparison.ipynb                  # Comprehensive model comparison
│   └── models_performance_analysis.ipynb         # Model comparison and performance analysis
│
├── models/                                        # Saved model files
├── metrics/                                       # Model evaluation metrics (csv files)
├── plots/                                         # Generated visualization plots
└── README.md                                      # Project documentation
```

## Features

- **Comprehensive Data Analysis**: Detailed exploratory data analysis with statistical insights and visualizations
- **Dimensionality Reduction**: Principal Component Analysis (PCA) for feature optimization
- **Multiple Machine Learning Approaches**: 
  - Tree-based models: Decision Tree, Random Forest
  - Gradient boosting method: XGBoost
  - Advanced ensemble methods: TabPFN
  - Neural Networks: Multi-layer Perceptron, Neural Network TensorFlow Keras

- **Model Comparison**: Systematic evaluation and comparison of all models
- **Performance Visualization**: Interactive plots and comprehensive metrics analysis
- **Automated Metrics**: R², MSE, MAE, MSLE, MRSE, and training time tracking

## Models Implemented

### 1. Tree-Based Models
- **Decision Tree**: Simple interpretable regression tree
- **Random Forest**: Ensemble of decision trees
- **XGBoost**: Gradient boosting with advanced optimization

### 2. Neural Networks
- **Multi-layer Perceptron (MLP)**: Traditional feedforward neural network
- **Keras Neural Network**: Deep learning model using TensorFlow Keras neural network

### 3. Advanced Methods
- **TabPFN**: Prior-fitted Networks for tabular data (state-of-the-art)

### 4. Analysis Tools
- **Principal Component Analysis (PCA)**: Dimensionality reduction and feature analysis
- **Comprehensive Metrics Comparison**: Automated model evaluation and ranking

## Requirements

```bash
# Core dependencies
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0

# Machine Learning libraries
xgboost>=1.5.0
tensorflow>=2.8.0
tabpfn>=0.1.0

# Visualization and analysis
plotly>=5.0.0
tabulate>=0.8.0

# Jupyter environment
jupyter>=1.0.0
ipykernel>=6.0.0
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/XxkmoxX/ML_geoTemp_prediction.git
cd ML_geoTemp_prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter:
```bash
jupyter notebook
```

## Usage

### Step-by-Step Analysis

1. **Exploratory Data Analysis**: Start with `EDA_geoTemp_prediction.ipynb`
   - Understand data distribution and patterns
   - Identify correlations and outliers
   - Generate statistical summaries

2. **Dimensionality Analysis**: Run `PCA_geoTemp_prediction.ipynb`
   - Perform Principal Component Analysis
   - Evaluate feature importance
   - Optimize feature selection

3. **Model Training**: Execute the regression notebooks in order:
   - `ML_regression1_geoTemp_prediction.ipynb` - Tree-based, gradient boosting (XGBoost) and advanced method (TabPFN).
   - `ML_regression2_geoTemp_prediction.ipynb` - Neural networks

4. **Model Comparison**: Use comparison notebooks for evaluation:
   - `metrics_comparison.ipynb` - Compare all model metrics
   - `models_performance_analysis.ipynb` - Detailed performance analysis

### Quick Start

For a quick overview of model performance:
```bash
# Run the performance analysis notebook directly
jupyter notebook models_performance_analysis.ipynb
```

## Results

The project provides comprehensive evaluation of various machine learning approaches for geothermal temperature prediction:

### Model Performance Metrics
- **R² Score**: Coefficient of determination for model accuracy
- **MSE**: Mean Squared Error for prediction quality
- **MAE**: Mean Absolute Error for average prediction deviation
- **Training Time**: Computational efficiency comparison
- **Efficiency Ratio**: R² score per second of training time

### Key Findings
- Systematic comparison of traditional ML vs. modern deep learning approaches
- Performance vs. computational cost trade-off analysis
- Feature importance insights through PCA analysis
- Optimal model selection based on multiple evaluation criteria

### Visualization Outputs
- Comprehensive model comparison charts
- Performance vs. training time scatter plots
- Error metrics comparison visualizations
- Efficiency analysis rankings

All results are automatically saved in the `metrics/` directory and visualizations in the `plots/` directory.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the geochemical research community for data insights
- Inspired by advances in machine learning for geothermal energy applications
- Built with open-source machine learning libraries

## Contact

- **Author**: Cristian Picighelli
- **Repository**: [ML_geoTemp_prediction](https://github.com/XxkmoxX/ML_geoTemp_prediction)
- **Issues**: Please report bugs and feature requests through GitHub Issues
