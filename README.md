# House Price Prediction

A complete machine learning project for predicting house prices using the California Housing dataset.

## Features

### Data Analysis
- Exploratory Data Analysis (EDA) with interactive visualizations
- Correlation heatmaps and feature analysis
- Geographic price distribution maps
- Missing value handling and data cleaning

### Machine Learning Models
- Ridge Regression
- Lasso Regression
- Random Forest
- Gradient Boosting
- XGBoost
- Cross-validation with 5-fold CV
- Hyperparameter comparison

### Feature Engineering
- Rooms per household
- Bedrooms per room ratio
- Population per household
- Income bracket categorization

### Model Evaluation
- R² Score, RMSE, MAE, MAPE metrics
- Cross-validation scores with standard deviation
- Actual vs Predicted analysis
- Residual distribution plots
- Feature importance analysis

### Interactive Dashboard
- **Data Explorer** - Filter and visualize housing data
- **Predict Price** - Single property price prediction
- **Batch Prediction** - Upload CSV for bulk predictions with download
- **Model Insights** - Compare models and view feature importance
- Sidebar with dataset info and model metrics
- Custom CSS styling

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run the Dashboard
```bash
python -m streamlit run app.py
```

### Run the Notebook
```bash
jupyter notebook HOUSE_PRICE_PREDICTION.ipynb
```

## Project Structure

```
├── housing.csv                    # California Housing dataset
├── HOUSE_PRICE_PREDICTION.ipynb   # Analysis & training notebook
├── app.py                         # Streamlit dashboard
├── requirements.txt               # Python dependencies
├── README.md                      # Documentation
├── best_house_price_model.pkl     # Trained model (auto-generated)
└── model_features.pkl             # Model metadata (auto-generated)
```

## How It Works

1. The app loads the housing dataset automatically
2. If no trained model exists, it trains one on first launch
3. The best model is selected automatically based on R² score
4. Use the dashboard to explore data, make predictions, or upload batch CSVs

## Dashboard Tabs

| Tab | Description |
|-----|-------------|
| Data Explorer | Interactive data filtering, distributions, scatter plots |
| Predict Price | Adjust sliders and inputs to predict a single house price |
| Batch Prediction | Upload CSV file, get predictions, download results |
| Model Insights | Model comparison charts, feature correlations, statistics |

## Technologies

- Python 3.10+
- Pandas, NumPy
- Scikit-learn, XGBoost
- Streamlit, Plotly
- Matplotlib, Seaborn
- Joblib
