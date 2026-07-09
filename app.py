import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv('housing.csv')

def train_model(df):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import GradientBoostingRegressor
    
    df['rooms_per_household'] = df['total_rooms'] / df['households']
    df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
    df['population_per_household'] = df['population'] / df['households']
    
    X = df.drop('median_house_value', axis=1)
    y = df['median_house_value']
    
    numeric_features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                        'total_bedrooms', 'population', 'households', 'median_income',
                        'rooms_per_household', 'bedrooms_per_room', 'population_per_household']
    categorical_features = ['ocean_proximity']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, 'best_house_price_model.pkl')
    joblib.dump(list(X.columns), 'model_features.pkl')
    return pipeline, list(X.columns)

@st.cache_resource
def load_model():
    df = load_data()
    if os.path.exists('best_house_price_model.pkl') and os.path.exists('model_features.pkl'):
        try:
            model = joblib.load('best_house_price_model.pkl')
            features = joblib.load('model_features.pkl')
            return model, features
        except:
            pass
    model, features = train_model(df)
    return model, features

df = load_data()
model, feature_names = load_model()

st.title("🏠 House Price Prediction Dashboard")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Data Explorer", "🤖 Predict Price", "📈 Model Insights"])

with tab1:
    st.header("Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Features", df.shape[1] - 1)
    col3.metric("Avg Price", f"${df['median_house_value'].mean():,.0f}")
    col4.metric("Median Price", f"${df['median_house_value'].median():,.0f}")
    
    st.subheader("House Value Distribution")
    fig = px.histogram(df, x='median_house_value', nbins=50, 
                       title='Distribution of House Values',
                       color_discrete_sequence=['#636EFA'])
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Income vs House Value")
        fig = px.scatter(df, x='median_income', y='median_house_value',
                        color='ocean_proximity', opacity=0.5,
                        title='Income vs House Value by Ocean Proximity')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Geographic Distribution")
        fig = px.scatter_mapbox(df, lat='latitude', lon='longitude',
                               color='median_house_value', size='population',
                               color_continuous_scale='Viridis',
                               mapbox_style='open-street-map',
                               title='House Prices by Location',
                               zoom=5, opacity=0.5)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Predict House Price")
    
    if model is None:
        st.warning("Model training failed. Please check the data.")
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            longitude = st.slider("Longitude", float(df['longitude'].min()), 
                                  float(df['longitude'].max()), float(df['longitude'].mean()))
            latitude = st.slider("Latitude", float(df['latitude'].min()),
                                float(df['latitude'].max()), float(df['latitude'].mean()))
            housing_age = st.slider("Housing Median Age", int(df['housing_median_age'].min()),
                                   int(df['housing_median_age'].max()), int(df['housing_median_age'].median()))
        
        with col2:
            total_rooms = st.number_input("Total Rooms", int(df['total_rooms'].min()),
                                         int(df['total_rooms'].max()), int(df['total_rooms'].median()))
            total_bedrooms = st.number_input("Total Bedrooms", int(df['total_bedrooms'].min()),
                                            int(df['total_bedrooms'].max()), int(df['total_bedrooms'].median()))
            population = st.number_input("Population", int(df['population'].min()),
                                        int(df['population'].max()), int(df['population'].median()))
        
        with col3:
            households = st.number_input("Households", int(df['households'].min()),
                                        int(df['households'].max()), int(df['households'].median()))
            median_income = st.slider("Median Income (in tens of thousands)", 
                                     float(df['median_income'].min()),
                                     float(df['median_income'].max()),
                                     float(df['median_income'].median()))
            ocean_proximity = st.selectbox("Ocean Proximity", df['ocean_proximity'].unique())
        
        if st.button("🔮 Predict Price", type="primary"):
            rooms_per_household = total_rooms / households if households > 0 else 0
            bedrooms_per_room = total_bedrooms / total_rooms if total_rooms > 0 else 0
            pop_per_household = population / households if households > 0 else 0
            
            input_data = pd.DataFrame({
                'longitude': [longitude],
                'latitude': [latitude],
                'housing_median_age': [housing_age],
                'total_rooms': [total_rooms],
                'total_bedrooms': [total_bedrooms],
                'population': [population],
                'households': [households],
                'median_income': [median_income],
                'ocean_proximity': [ocean_proximity]
            })
            
            prediction = model.predict(input_data)[0]
            
            st.markdown("---")
            st.subheader("Prediction Result")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted Price", f"${prediction:,.0f}")
            col2.metric("Min Area Price", f"${df['median_house_value'].min():,.0f}")
            col3.metric("Max Area Price", f"${df['median_house_value'].max():,.0f}")
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prediction,
                title = {'text': "Predicted House Price"},
                gauge = {
                    'axis': {'range': [df['median_house_value'].min(), df['median_house_value'].max()]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, df['median_house_value'].quantile(0.25)], 'color': "lightgreen"},
                        {'range': [df['median_house_value'].quantile(0.25), df['median_house_value'].quantile(0.75)], 'color': "yellow"},
                        {'range': [df['median_house_value'].quantile(0.75), df['median_house_value'].max()], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': prediction
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Model Insights")
    
    st.subheader("Feature Correlations with House Value")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_cols].corr()['median_house_value'].sort_values(ascending=False)
    
    fig = px.bar(x=correlations.index[1:], y=correlations.values[1:],
                title='Feature Correlations with House Value',
                labels={'x': 'Feature', 'y': 'Correlation'},
                color=correlations.values[1:],
                color_continuous_scale='RdYlGn')
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Ocean Proximity Analysis")
    ocean_stats = df.groupby('ocean_proximity')['median_house_value'].agg(['mean', 'median', 'std']).reset_index()
    
    fig = px.bar(ocean_stats, x='ocean_proximity', y=['mean', 'median'],
                title='Average House Value by Ocean Proximity',
                barmode='group')
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Feature Statistics")
    st.dataframe(df.describe().style.format(precision=2))
