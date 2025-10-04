import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utilsforecast.plotting import plot_series
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import *

# needed for baseline models
# they have best implementations of statistical forecasting models
from statsforecast import StatsForecast
from statsforecast.models import Naive, HistoricAverage, WindowAverage, SeasonalNaive

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