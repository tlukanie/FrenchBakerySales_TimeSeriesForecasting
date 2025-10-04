import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utilsforecast.plotting import plot_series
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import *

# line 68 - 180 is a Baseline model
# line 182 ARIMA model
# line 335 cross-validation

# needed for baseline models
# they have best implementations of statistical forecasting models
from statsforecast import StatsForecast
from statsforecast.models import Naive, HistoricAverage, WindowAverage, SeasonalNaive


from statsforecast.models import AutoARIMA #for ARIMA model

import warnings
warnings.filterwarnings("ignore")

#intitial set up
df = pd.read_csv("data/daily_sales_french_bakery.csv", parse_dates=["ds"])
# test
print("Unique product IDs in the dataset:")
print(df['unique_id'].unique())

# After loading the data
baguette_data = df[df['unique_id'] == 'BAGUETTE']
croissant_data = df[df['unique_id'] == 'CROISSANT']
print(f"BAGUETTE data points: {len(baguette_data)}")
print(f"CROISSANT data points: {len(croissant_data)}")


df = df.groupby('unique_id').filter(lambda x: len(x) >= 28)
df = df.drop(["unit_price"], axis=1) #removing the last column to make it easier working with dataset
print(df.head())

# visualization for all time data
plt.figure(figsize=(12, 6))
for product_id in ['BAGUETTE', 'CROISSANT']:
    product_data = df[df['unique_id'] == product_id]
    if not product_data.empty:
        plt.plot(product_data['ds'], product_data['y'], label=product_id)
plt.legend()
plt.title('Full Sales Data Time Series')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#visualization for last 56 days
plt.figure(figsize=(12, 6))
for product_id in ['BAGUETTE', 'CROISSANT']:
    product_data = df[df['unique_id'] == product_id]
    if not product_data.empty:
        # Only show the most recent 56 data points (similar to max_insample_length=56)
        recent_data = product_data.tail(56)
        plt.plot(recent_data['ds'], recent_data['y'], label=product_id)
plt.legend()
plt.title('Recent 56 Days Sales Data')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# Baseline Models
horizon = 7 # to forecast the dta for the next 7 days

models = [
    Naive(),
    HistoricAverage(),
    WindowAverage(window_size=7), # taking average of the last 7 days and forecasting it into the future
    SeasonalNaive(season_length=7) # seasonality of 1 week
]

sf = StatsForecast(models=models, freq="D") # forecasting object responsible for training, prediction, etc, freq stands for the daily frequency with D
sf.fit(df=df)
preds = sf.predict(h=horizon)
print(preds.head())


# First, let's inspect the structure of the predictions dataframe
print("Prediction dataframe columns:")
print(preds.columns)
print("Sample prediction data:")
print(preds.head())

# plotting the predictions using matplotlib
for product_id in ['BAGUETTE', 'CROISSANT']:
    plt.figure(figsize=(14, 7))
    
    # Get historical data (last 28 points only)
    product_data = df[df['unique_id'] == product_id]
    if product_data.empty:
        continue
    
    historical_data = product_data.tail(28)
    plt.plot(historical_data['ds'], historical_data['y'], 'b-', label='Historical Data')
    
    # Get predictions for this product
    product_preds = preds[preds['unique_id'] == product_id]
    
    if product_preds.empty:
        print(f"No predictions found for {product_id}")
        continue
    
    # Find the last historical date to align forecasts
    last_date = historical_data['ds'].max()
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq='D')
    
    # Plot each model's predictions - adapt to the actual column names
    model_names = ['Naive', 'HistoricAverage', 'WindowAverage', 'SeasonalNaive']
    colors = ['red', 'green', 'orange', 'purple']
    
    # Try different possible column name formats
    for i, model_name in enumerate(model_names):
        # Check different possible column formats
        possible_columns = [
            f'{model_name}-mean',  # Standard format
            model_name,            # Simple format
            model_name.lower(),    # Lowercase
            f'{model_name.lower()}-mean'  # Lowercase with -mean
        ]
        
        column_found = False
        for col in possible_columns:
            if col in product_preds.columns:
                plt.plot(forecast_dates, product_preds[col], 
                        color=colors[i], linestyle='--', marker='o', label=model_name)
                column_found = True
                break
        
        if not column_found:
            print(f"No prediction column found for model {model_name}")
    
    plt.title(f'Forecast for {product_id} (Last 28 days + {horizon} days forecast)')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()
    

# Evaluate baseline models
test = df.groupby("unique_id").tail(7)
train = df.drop(test.index).reset_index(drop=True)

sf.fit(df=train)
preds = sf.predict(h=horizon)
eval_df = pd.merge(test, preds, 'left', ['ds', 'unique_id'])

evaluation = evaluate(
    eval_df,
    metrics=[mae], # mean absolute error (average of the absolute distance between the predictions and the actual values)
)
print("Evaluation head")
print(evaluation.head())
# transforms the detailed per product evaluation metrics into average metrix across all products
evaluation = evaluation.drop(['unique_id'], axis=1).groupby('metric').mean().reset_index()
print(evaluation)

# visualization of evaluation
methods = evaluation.columns[1:].tolist()  
values = evaluation.iloc[0, 1:].tolist() 

plt.figure(figsize=(10, 6))
bars = plt.bar(methods, values)

for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
             f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

plt.xlabel('Methods')
plt.ylabel('Mean absolute error (MAE)')
plt.tight_layout()

plt.show()

# ARIMA
# AR: autoregressive model, I: order of integration, MA: moving average model
# ARIMA: a flexible model that can forecast series with a trend and one seasonl period
# ARMA can model only stationary time series
# Stationary series: Series with constant mean and variance (no trend, no sesonality)
# I - integration order, internally transforms the series to make it stationary (we don't have to transform data,
# ourselves, the model can take non-stationary data directly)
# SARIMA - seasonal ARIMA model
# In practice we only have to set m (the length of  season)
# AutoARIMA
print("*********ARIMA*********")
unique_ids = ["BAGUETTE", "CROISSANT"]

small_train = train[train["unique_id"].isin(unique_ids)]
small_test = test[test["unique_id"].isin(unique_ids)]

models = [
    AutoARIMA(seasonal=False, alias="ARIMA"),
    AutoARIMA(season_length=7, alias="SARIMA")
]

sf = StatsForecast(models=models, freq="D")
sf.fit(df=small_train)
arima_preds = sf.predict(h=horizon)

arima_eval_df = pd.merge(arima_preds, eval_df, 'inner', ['ds', 'unique_id'])
arima_eval = evaluate(
    arima_eval_df,
    metrics=[mae],
)
print(arima_eval)
arima_eval = arima_eval.drop(['unique_id'], axis=1).groupby('metric').mean().reset_index()
print(arima_eval)

# Visualize ARIMA predictions with matplotlib
for product_id in ['BAGUETTE', 'CROISSANT']:
    plt.figure(figsize=(14, 7))
    
    # Get historical data (last 28 points only)
    product_data = small_train[small_train['unique_id'] == product_id]
    if product_data.empty:
        continue
    
    historical_data = product_data.tail(28)
    plt.plot(historical_data['ds'], historical_data['y'], 'b-', label='Historical Data')
    
    # Get actual test data
    test_data = small_test[small_test['unique_id'] == product_id]
    if not test_data.empty:
        plt.plot(test_data['ds'], test_data['y'], 'k--', marker='o', label='Actual Values')
    
    # Get ARIMA predictions for this product
    product_preds = arima_preds[arima_preds['unique_id'] == product_id]
    
    if product_preds.empty:
        print(f"No ARIMA predictions found for {product_id}")
        continue
    
    # Find the last historical date to align forecasts
    last_date = historical_data['ds'].max()
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq='D')
    
    # Plot each ARIMA model's predictions - using actual column names
    colors = ['red', 'purple']
    for i, model_name in enumerate(['ARIMA', 'SARIMA']):
        if model_name in product_preds.columns:
            plt.plot(forecast_dates, product_preds[model_name], 
                    color=colors[i], linestyle='--', marker='o', label=model_name)
        else:
            print(f"Column {model_name} not found in predictions. Available columns: {product_preds.columns}")
    
    plt.title(f'ARIMA Forecast for {product_id} (Last 28 days + {horizon} days forecast)')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Compare all models together
for product_id in ['BAGUETTE', 'CROISSANT']:
    plt.figure(figsize=(14, 7))
    
    # Get test data for actual values
    test_data = small_test[small_test['unique_id'] == product_id]
    if test_data.empty:
        continue
    
    plt.plot(test_data['ds'], test_data['y'], 'k-', marker='o', linewidth=2, label='Actual Values')
    
    # Get standard model predictions
    base_product_preds = preds[preds['unique_id'] == product_id]
    
    # Get ARIMA predictions
    arima_product_preds = arima_preds[arima_preds['unique_id'] == product_id]
    
    if base_product_preds.empty or arima_product_preds.empty:
        print(f"Missing predictions for {product_id}")
        continue
    
    # Print available columns to diagnose issues
    print(f"Base model columns for {product_id}: {base_product_preds.columns}")
    print(f"ARIMA model columns for {product_id}: {arima_product_preds.columns}")
    
    # Plot best baseline model - determine correct column names first
    baseline_models = ['SeasonalNaive', 'Naive', 'WindowAverage', 'HistoricAverage']
    baseline_col = None
    for model in baseline_models:
        if model in base_product_preds.columns:
            baseline_col = model
            break
    
    if baseline_col:
        plt.plot(test_data['ds'], base_product_preds[baseline_col], 'g--', marker='x', label=baseline_col)
    
    # Plot ARIMA models - using actual column names
    if 'ARIMA' in arima_product_preds.columns:
        plt.plot(test_data['ds'], arima_product_preds['ARIMA'], 'r--', marker='+', label='ARIMA')
    if 'SARIMA' in arima_product_preds.columns:
        plt.plot(test_data['ds'], arima_product_preds['SARIMA'], 'b--', marker='*', label='SARIMA')
    
    plt.title(f'Model Comparison for {product_id} ({horizon} Day Forecast)')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
# visualization of different models' mae
methods = arima_eval.columns[1:].tolist()  
values = arima_eval.iloc[0, 1:].tolist() 

sorted_data = sorted(zip(methods, values), key=lambda x: x[1], reverse=True)
methods_sorted, values_sorted = zip(*sorted_data)

plt.figure(figsize=(10, 6))
bars = plt.bar(methods_sorted, values_sorted)

for bar, value in zip(bars, values_sorted):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
             f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

plt.xlabel('Methods')
plt.ylabel('Mean absolute error (MAE)')
plt.tight_layout()

plt.show()


# Cross-validation
# Evaluating on a single forecast period is not enough (1. Few datapoints, 2. Not representative of the model's
# forecasting capability)
# Use cross-validation to forecast on many windows for a more reliable evaluation
small_df = df[df["unique_id"].isin(unique_ids)]

models = [
    SeasonalNaive(season_length=7),
    AutoARIMA(seasonal=False, alias="ARIMA"),
    AutoARIMA(season_length=7, alias="SARIMA")
]

sf = StatsForecast(models=models, freq="D")
cv_df = sf.cross_validation(
    h=horizon, # 7 days
    df=small_df,
    n_windows=8, # create 8 different testing windows
    step_size=horizon, # move forward by 7 days each time
    refit=True # retrain models for each window
)

print(cv_df.head())

# visualization