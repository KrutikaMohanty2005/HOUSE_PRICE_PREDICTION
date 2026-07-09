# 🏠 House Price Prediction - Upgraded Version

A complete machine learning project for predicting house prices using the California Housing dataset.

## Features

### 📊 Data Analysis
- Exploratory Data Analysis (EDA)
- Correlation heatmaps
- Geographic visualizations

### 🤖 Machine Learning Models
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest
- Gradient Boosting
- XGBoost

### 🔧 Feature Engineering
- Rooms per household
- Bedrooms per room ratio
- Population per household
- Income categories

### 📈 Model Comparison
- RMSE, MAE, R² metrics
- Cross-validation scores
- Feature importance analysis

### 🌐 Interactive Dashboard
- Streamlit web application
- Real-time price predictions
- Data exploration tools
- Model insights visualization

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Run the Notebook
```bash
jupyter notebook HOUSE_PRICE_PREDICTION.ipynb
```

### 2. Launch Dashboard
```bash
streamlit run app.py
```

## Project Structure

```
├── housing.csv                    # Dataset
├── HOUSE_PRICE_PREDICTION.ipynb   # Main analysis notebook
├── app.py                         # Streamlit dashboard
├── requirements.txt               # Python dependencies
├── best_house_price_model.pkl     # Trained model (generated)
├── model_features.pkl             # Feature list (generated)
└── README.md                      # This file
```

## Model Performance

| Model | RMSE | R² Score |
|-------|------|----------|
| Linear Regression | - | - |
| Ridge Regression | - | - |
| Lasso Regression | - | - |
| Random Forest | - | - |
| Gradient Boosting | - | - |
| XGBoost | - | - |

*Run the notebook to see actual performance metrics*

## Technologies Used

- Python 3.8+
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Streamlit
- Plotly
