# model/utils.py
import pandas as pd

def preprocess_input(input_df, scaler, encoder_columns):
    # Scale numeric columns
    numeric_cols = ['Monthly Charges', 'Total Charges', 'Tenure Months', 'CLTV', 'TotalServicesOpted']
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # One-hot encoded columns might be missing, ensure all exist
    for col in encoder_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match training
    input_df = input_df[encoder_columns]

    return input_df
