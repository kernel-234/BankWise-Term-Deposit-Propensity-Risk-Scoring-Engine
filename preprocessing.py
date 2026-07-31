import pandas as pd
import numpy as np
import logging

# Set up logging to track operations in production
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def clean_banking_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executes systematic data cleaning on the raw banking dataset:
    1. Removes duplicate rows
    2. Drops redundant/composite columns (marital_status, day_month)
    3. Imputes explicit NaNs
    4. Maps target variable 'y' to binary integers (1/0)
    5. Strips whitespace from string columns
    """
    logging.info(f"Starting data cleaning. Initial dataset shape: {df.shape}")
    
    # Step 1: Remove exact duplicates
    initial_count = len(df)
    df = df.drop_duplicates().copy()
    logging.info(f"Dropped {initial_count - len(df)} exact duplicate rows.")
    
    # Step 2: Drop redundant and composite columns
    cols_to_drop = []
    if 'marital_status' in df.columns and 'marital' in df.columns:
        cols_to_drop.append('marital_status')
    if 'day_month' in df.columns:
        cols_to_drop.append('day_month')
        
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logging.info(f"Dropped redundant columns: {cols_to_drop}")
        
    # Step 3: Handle explicit NaNs
    if 'marital' in df.columns and df['marital'].isna().sum() > 0:
        df['marital'] = df['marital'].fillna(df['marital'].mode()[0])
    if 'education' in df.columns and df['education'].isna().sum() > 0:
        df['education'] = df['education'].fillna('unknown')
        
    # Step 4: Standardize Target Variable 'y'
    if 'y' in df.columns:
        df['y'] = df['y'].map({'yes': 1, 'no': 0}).astype(int)
        logging.info("Target 'y' mapped successfully: 'yes' -> 1, 'no' -> 0.")
        
    # Step 5: Clean whitespace across string features
    string_cols = df.select_dtypes(include=['object']).columns
    for col in string_cols:
        df[col] = df[col].astype(str).str.strip()
        
    logging.info(f"Cleaning complete. Final dataset shape: {df.shape}")
    return df

if __name__ == "__main__":
    # Test script standalone
    raw_path = "data/raw/banking_data.csv"
    processed_path = "data/processed/banking_cleaned.csv"
    
    logging.info(f"Loading raw data from {raw_path}")
    raw_df = pd.read_csv(raw_path)
    
    cleaned_df = clean_banking_data(raw_df)
    
    cleaned_df.to_csv(processed_path, index=False)
    logging.info(f"Saved cleaned data to {processed_path}")