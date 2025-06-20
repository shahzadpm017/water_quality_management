#!/usr/bin/env python
# coding: utf-8

"""
Water Quality Prediction Script
- Loads and explores water quality data
- Prepares features
- (Add modeling and prediction steps as needed)
"""

# --- Import Libraries ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

# --- Load Data ---
df = pd.read_csv('PB_All_2000_2021.csv', sep=';')

# --- Basic Exploration ---
print("\nData Info:")
df.info()
print("\nData Shape:", df.shape)
print("\nData Description:")
print(df.describe().T)
print("\nMissing Values:")
print(df.isnull().sum())

# --- Date Processing ---
df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
df = df.sort_values(by=['id', 'date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

# --- Show Columns ---
print("\nColumns:", df.columns.tolist())

# --- Select Target Pollutants ---
pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']

# --- (Continue with feature engineering, modeling, etc. as needed) ---

