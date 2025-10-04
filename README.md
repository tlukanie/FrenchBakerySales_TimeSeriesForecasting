# French Bakery Time Series Forecasting
Source: https://www.youtube.com/watch?v=fxx_E0ojKrc </br>
Streamlit tutorial: https://www.youtube.com/watch?v=d7fnzDQ5qM8 </br>

## Project Overview
This project implements time series forecasting for a French bakery's product sales. It uses several statistical models to predict daily product sales, focusing on BAGUETTE and CROISSANT items, with a 7-day forecast horizon.

## Current Implementation

### 1. Data Preparation and Exploration
- **Data Loading**: Loads sales data from CSV file with dates parsed correctly
- **Filtering**: Includes only products with at least 28 data points
- **Preprocessing**: Removes unnecessary columns like unit_price
- **Visualization**: Creates time series plots of full historical data and recent 56 days

### 2. Baseline Models
- **Models Implemented**:
  - `Naive`: Uses last observed value for all future predictions
  - `HistoricAverage`: Uses mean of all historical values
  - `WindowAverage`: Uses mean of last 7 days
  - `SeasonalNaive`: Uses value from 7 days ago (captures weekly seasonality)
- **Training and Prediction**: Uses StatsForecast package to fit models and generate forecasts
- **Visualization**: Plots historical data and forecasts for each product

### 3. ARIMA Models
- **Models Implemented**:
  - `ARIMA`: Standard AutoARIMA model without seasonal component
  - `SARIMA`: Seasonal AutoARIMA with 7-day seasonality
- **Training and Prediction**: Fits models on training data, generates 7-day forecasts
- **Visualization**: Plots historical data, actual test values, and forecasts

### 4. Model Evaluation
- **Methodology**: 
  - Single train-test split with last 7 days as test data
  - Cross-validation with 8 windows for more reliable evaluation
- **Metrics**: Mean Absolute Error (MAE)
- **Visualization**: Bar charts comparing MAE across different models

### 5. Advanced Evaluation: Cross-Validation
- **Implementation**: 8 different testing windows with 7-day step size
- **Model Retraining**: Models retrained for each window
- **Purpose**: More robust evaluation across different time periods

## Future Roadmap: Streamlit Interactive App

### 1. App Structure and Features

#### Home Page
- **Overview Dashboard**:
  - Summary statistics of bakery sales
  - Interactive time series plot of historical sales
  - Calendar heatmap showing daily/weekly patterns

#### Forecasting Section
- **Model Selection**:
  - Dropdown for selecting forecasting models (Naive, ARIMA, SARIMA, etc.)
  - Option to combine multiple models
  
- **Product Selection**:
  - Multi-select for choosing which bakery products to forecast
  - Option to group similar products

- **Forecasting Parameters**:
  - Slider for forecast horizon (1-30 days)
  - Calendar picker for specific forecast start date
  - Advanced options panel for model-specific parameters

#### Results Visualization
- **Interactive Plots**:
  - Forecast plots with confidence intervals
  - Actual vs. predicted comparisons
  - Model performance metrics (MAE, RMSE)
  
- **Scenario Analysis**:
  - "What-if" analysis for special events/holidays
  - Anomaly detection for unusual sales patterns

#### Model Evaluation
- **Performance Metrics**:
  - Cross-validation results visualization
  - Model comparison dashboard
  - Error analysis by day of week, product type

### 2. Technical Implementation

#### Backend Requirements
- Store trained models for quick loading
- Implement periodic retraining schedule
- Add data validation and preprocessing pipeline

#### Frontend Features
- Responsive design for desktop and tablet
- Dark/light mode toggle
- Export capabilities (CSV, PDF reports)
- User authentication for bakery staff

### 3. Development Phases

#### Phase 1: Core Functionality
- Basic app with model selection and forecasting
- Simple visualization of results
- Product filtering

#### Phase 2: Enhanced Features
- Cross-validation visualization
- Confidence intervals for forecasts
- Performance metrics dashboard

#### Phase 3: Advanced Analytics
- Anomaly detection
- Special event handling
- Sales pattern insights
- Inventory optimization recommendations

### 4. Deployment Considerations
- Cloud hosting options (Streamlit Cloud, Heroku)
- Data security and privacy
- Regular model retraining pipeline
- Performance optimization for large datasets

## Conclusion
The current implementation provides solid foundations for time series forecasting of bakery sales. The Streamlit app will make this functionality accessible to non-technical users, enabling data-driven inventory planning and business decision making.
