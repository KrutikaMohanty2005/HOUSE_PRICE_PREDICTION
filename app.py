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

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1f77b4; text-align: center; padding: 1rem 0; }
    .sub-header { font-size: 1.2rem; color: #555; text-align: center; margin-bottom: 2rem; }
    .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 12px; color: white; text-align: center; }
    .metric-value { font-size: 1.8rem; font-weight: 700; }
    .metric-label { font-size: 0.9rem; opacity: 0.9; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 10px 24px; border-radius: 8px 8px 0 0; }
    div[data-testid="stMetric"] { background-color: #f0f2f6; padding: 12px; border-radius: 8px; border-left: 4px solid #1f77b4; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv('housing.csv')
    df['total_bedrooms'].fillna(df['total_bedrooms'].median(), inplace=True)
    df['rooms_per_household'] = df['total_rooms'] / df['households']
    df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
    df['population_per_household'] = df['population'] / df['households']
    return df


def train_model(df):
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    X = df.drop('median_house_value', axis=1)
    y = df['median_house_value']

    numeric_features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                        'total_bedrooms', 'population', 'households', 'median_income',
                        'rooms_per_household', 'bedrooms_per_room', 'population_per_household']
    categorical_features = ['ocean_proximity']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Random Forest': RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=150, random_state=42),
    }

    best_score = -1
    best_model = None
    best_name = ""
    results = {}

    for name, m in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', m)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        cv = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')

        results[name] = {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'CV_Mean': cv.mean(), 'CV_Std': cv.std()}

        if r2 > best_score:
            best_score = r2
            best_model = pipeline
            best_name = name

    joblib.dump(best_model, 'best_house_price_model.pkl')
    joblib.dump({'features': list(X.columns), 'numeric_features': numeric_features,
                 'categorical_features': categorical_features, 'results': results,
                 'best_name': best_name, 'metrics': results[best_name]}, 'model_features.pkl')

    return best_model, list(X.columns), results, best_name, results[best_name]


@st.cache_resource
def load_model():
    df = load_data()
    if os.path.exists('best_house_price_model.pkl') and os.path.exists('model_features.pkl'):
        try:
            model = joblib.load('best_house_price_model.pkl')
            meta = joblib.load('model_features.pkl')
            if isinstance(meta, dict):
                return (model, meta.get('features', []), meta.get('results', {}),
                        meta.get('best_name', ''), meta.get('metrics', {}))
            return model, meta, {}, '', {}
        except Exception:
            pass
    return train_model(df)


df = load_data()
model, feature_names, model_results, best_model_name, best_metrics = load_model()

with st.sidebar:
    st.image("https://img.icons8.com/color/96/real-estate.png", width=64)
    st.title("House Price Predictor")
    st.markdown("---")
    st.markdown("### Dataset Info")
    st.info(f"**Records:** {len(df):,}\n\n**Features:** {df.shape[1] - 1}\n\n**Avg Price:** ${df['median_house_value'].mean():,.0f}")
    st.markdown("---")
    st.markdown("### Model Info")
    if best_model_name:
        st.success(f"**Best Model:** {best_model_name}")
        if best_metrics:
            st.metric("R² Score", f"{best_metrics.get('R2', 0):.4f}")
            st.metric("RMSE", f"${best_metrics.get('RMSE', 0):,.0f}")
    st.markdown("---")
    st.caption("Built with Streamlit + Scikit-learn")

st.markdown('<p class="main-header">House Price Prediction Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Explore housing data and predict property values using machine learning</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Data Explorer", "Predict Price", "Batch Prediction", "Model Insights"])

with tab1:
    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Avg Price", f"${df['median_house_value'].mean():,.0f}")
    with col3:
        st.metric("Median Price", f"${df['median_house_value'].median():,.0f}")
    with col4:
        st.metric("Price Std Dev", f"${df['median_house_value'].std():,.0f}")

    st.markdown("---")

    with st.expander("Filter Data", expanded=False):
        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            ocean_options = df['ocean_proximity'].unique().tolist()
            selected_ocean = st.multiselect("Ocean Proximity", ocean_options, default=ocean_options)
        with fcol2:
            income_range = st.slider("Income Range", float(df['median_income'].min()),
                                     float(df['median_income'].max()),
                                     (float(df['median_income'].min()), float(df['median_income'].max())))
        with fcol3:
            price_range = st.slider("Price Range", float(df['median_house_value'].min()),
                                    float(df['median_house_value'].max()),
                                    (float(df['median_house_value'].min()), float(df['median_house_value'].max())))

    filtered_df = df[
        (df['ocean_proximity'].isin(selected_ocean)) &
        (df['median_income'] >= income_range[0]) & (df['median_income'] <= income_range[1]) &
        (df['median_house_value'] >= price_range[0]) & (df['median_house_value'] <= price_range[1])
    ]

    st.info(f"Showing {len(filtered_df):,} of {len(df):,} records")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(filtered_df, x='median_house_value', nbins=50,
                           title='House Value Distribution', color_discrete_sequence=['#636EFA'])
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(filtered_df, x='ocean_proximity', y='median_house_value',
                     title='Price by Ocean Proximity', color='ocean_proximity')
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(filtered_df, x='median_income', y='median_house_value',
                         color='ocean_proximity', opacity=0.4, title='Income vs House Value',
                         trendline='ols')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.scatter(filtered_df, x='latitude', y='longitude',
                         color='median_house_value', size='population',
                         color_continuous_scale='Viridis', opacity=0.5,
                         title='Geographic Price Distribution')
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Raw Data Preview"):
        st.dataframe(filtered_df.head(100).style.format(precision=2), use_container_width=True)

with tab2:
    st.header("Predict House Price")

    if model is None:
        st.warning("Model training failed. Please check the data.")
    else:
        st.markdown("Adjust the parameters below and click **Predict** to estimate a house price.")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Location")
            longitude = st.slider("Longitude", float(df['longitude'].min()),
                                  float(df['longitude'].max()), float(df['longitude'].mean()), step=0.01)
            latitude = st.slider("Latitude", float(df['latitude'].min()),
                                 float(df['latitude'].max()), float(df['latitude'].mean()), step=0.01)
            ocean_proximity = st.selectbox("Ocean Proximity", df['ocean_proximity'].unique())

        with col2:
            st.subheader("Property Details")
            housing_age = st.slider("Housing Median Age", int(df['housing_median_age'].min()),
                                    int(df['housing_median_age'].max()), int(df['housing_median_age'].median()))
            total_rooms = st.number_input("Total Rooms", int(df['total_rooms'].min()),
                                          int(df['total_rooms'].max()), int(df['total_rooms'].median()))
            total_bedrooms = st.number_input("Total Bedrooms", int(df['total_bedrooms'].min()),
                                             int(df['total_bedrooms'].max()), int(df['total_bedrooms'].median()))

        with col3:
            st.subheader("Demographics")
            population = st.number_input("Population", int(df['population'].min()),
                                         int(df['population'].max()), int(df['population'].median()))
            households = st.number_input("Households", int(df['households'].min()),
                                         int(df['households'].max()), int(df['households'].median()))
            median_income = st.slider("Median Income (tens of thousands)",
                                      float(df['median_income'].min()),
                                      float(df['median_income'].max()),
                                      float(df['median_income'].median()), step=0.1)

        if st.button("Predict Price", type="primary", use_container_width=True):
            rooms_per_household = total_rooms / households if households > 0 else 0
            bedrooms_per_room = total_bedrooms / total_rooms if total_rooms > 0 else 0
            pop_per_household = population / households if households > 0 else 0

            input_data = pd.DataFrame({
                'longitude': [longitude], 'latitude': [latitude],
                'housing_median_age': [housing_age], 'total_rooms': [total_rooms],
                'total_bedrooms': [total_bedrooms], 'population': [population],
                'households': [households], 'median_income': [median_income],
                'ocean_proximity': [ocean_proximity],
                'rooms_per_household': [rooms_per_household],
                'bedrooms_per_room': [bedrooms_per_room],
                'population_per_household': [pop_per_household]
            })

            prediction = model.predict(input_data)[0]

            st.markdown("---")
            st.subheader("Prediction Result")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Predicted Price", f"${prediction:,.0f}")
            col2.metric("Min Area Price", f"${df['median_house_value'].min():,.0f}")
            col3.metric("Max Area Price", f"${df['median_house_value'].max():,.0f}")
            diff_pct = ((prediction - df['median_house_value'].mean()) / df['median_house_value'].mean()) * 100
            col4.metric("vs Average", f"{diff_pct:+.1f}%")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                title={'text': "Predicted House Price"},
                gauge={
                    'axis': {'range': [df['median_house_value'].min(), df['median_house_value'].max()]},
                    'bar': {'color': "#1f77b4"},
                    'steps': [
                        {'range': [0, df['median_house_value'].quantile(0.25)], 'color': "#d4edda"},
                        {'range': [df['median_house_value'].quantile(0.25), df['median_house_value'].quantile(0.75)], 'color': "#fff3cd"},
                        {'range': [df['median_house_value'].quantile(0.75), df['median_house_value'].max()], 'color': "#f8d7da"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': prediction
                    }
                }
            ))
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Input Summary"):
                st.json({
                    "Location": f"({latitude}, {longitude})",
                    "Ocean Proximity": ocean_proximity,
                    "Housing Age": housing_age,
                    "Rooms": total_rooms,
                    "Bedrooms": total_bedrooms,
                    "Population": population,
                    "Households": households,
                    "Median Income": f"${median_income * 10000:,.0f}",
                    "Derived Features": {
                        "Rooms/Household": round(rooms_per_household, 2),
                        "Bedrooms/Room": round(bedrooms_per_room, 3),
                        "Pop/Household": round(pop_per_household, 2)
                    }
                })

with tab3:
    st.header("Batch Prediction")
    st.markdown("Upload a CSV file with house features to get predictions for multiple properties.")

    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(batch_df)} records")

            required_cols = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                             'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity']
            missing = [c for c in required_cols if c not in batch_df.columns]

            if missing:
                st.error(f"Missing columns: {', '.join(missing)}")
                st.info(f"Required columns: {', '.join(required_cols)}")
            else:
                batch_df['rooms_per_household'] = batch_df['total_rooms'] / batch_df['households']
                batch_df['bedrooms_per_room'] = batch_df['total_bedrooms'] / batch_df['total_rooms']
                batch_df['population_per_household'] = batch_df['population'] / batch_df['households']

                predictions = model.predict(batch_df)
                batch_df['predicted_price'] = predictions

                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                col1.metric("Predictions Made", f"{len(predictions):,}")
                col2.metric("Avg Predicted Price", f"${predictions.mean():,.0f}")
                col3.metric("Price Range", f"${predictions.min():,.0f} - ${predictions.max():,.0f}")

                fig = px.histogram(batch_df, x='predicted_price', nbins=30,
                                   title='Predicted Price Distribution', color_discrete_sequence=['#2ecc71'])
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(batch_df.style.format({'predicted_price': '${:,.0f}',
                                                     'median_income': '{:.2f}',
                                                     'longitude': '{:.2f}',
                                                     'latitude': '{:.2f}'}), use_container_width=True)

                csv = batch_df.to_csv(index=False)
                st.download_button("Download Predictions", csv, "predictions.csv", "text/csv", use_container_width=True)

        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.info("Upload a CSV file with columns: " + ", ".join(['longitude', 'latitude', 'housing_median_age',
                'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity']))

        with st.expander("Download Sample Template"):
            sample = df[['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                         'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity']].head(5)
            csv = sample.to_csv(index=False)
            st.download_button("Download Template CSV", csv, "template.csv", "text/csv")

with tab4:
    st.header("Model Insights")

    if model_results:
        st.subheader("Model Comparison")

        results_df = pd.DataFrame(model_results).T
        results_df = results_df[['R2', 'RMSE', 'MAE', 'CV_Mean', 'CV_Std']]

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(results_df.reset_index(), x='index', y='R2',
                         title='R² Score Comparison', color='R2',
                         color_continuous_scale='Viridis')
            fig.update_layout(xaxis_title="Model", yaxis_title="R² Score")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(results_df.reset_index(), x='index', y='RMSE',
                         title='RMSE Comparison (Lower is Better)', color='RMSE',
                         color_continuous_scale='Reds_r')
            fig.update_layout(xaxis_title="Model", yaxis_title="RMSE ($)")
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(results_df.style.format({'R2': '{:.4f}', 'RMSE': '${:,.0f}',
                                                'MAE': '${:,.0f}', 'CV_Mean': '{:.4f}',
                                                'CV_Std': '{:.4f}'}), use_container_width=True)

    st.markdown("---")
    st.subheader("Feature Correlations with House Value")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_cols].corr()['median_house_value'].sort_values(ascending=False)

    fig = px.bar(x=correlations.index[1:], y=correlations.values[1:],
                 title='Feature Correlations with House Value',
                 labels={'x': 'Feature', 'y': 'Correlation'},
                 color=correlations.values[1:],
                 color_continuous_scale='RdYlGn')
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ocean Proximity Analysis")
        ocean_stats = df.groupby('ocean_proximity')['median_house_value'].agg(['mean', 'median', 'count']).reset_index()
        ocean_stats.columns = ['Ocean Proximity', 'Mean Price', 'Median Price', 'Count']
        fig = px.bar(ocean_stats, x='Ocean Proximity', y=['Mean Price', 'Median Price'],
                     title='Price by Ocean Proximity', barmode='group')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Price by Income Bracket")
        df['income_bracket'] = pd.cut(df['median_income'], bins=[0, 2, 4, 6, 8, np.inf],
                                       labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        income_stats = df.groupby('income_bracket', observed=True)['median_house_value'].mean().reset_index()
        fig = px.bar(income_stats, x='income_bracket', y='median_house_value',
                     title='Avg Price by Income Bracket', color='median_house_value',
                     color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Statistics")
    st.dataframe(df.describe().style.format(precision=2), use_container_width=True)
