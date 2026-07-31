import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies domain-driven feature engineering and encoding transformations.
    """
    df = df.copy()
    logging.info(f"Starting feature engineering on dataset with shape {df.shape}")

    # 1. Financial Debt Burden Feature
    df['total_debt_count'] = (
        (df['housing'].astype(str).str.lower() == 'yes').astype(int) + 
        (df['loan'].astype(str).str.lower() == 'yes').astype(int)
    )

    # 2. Non-linear Age Binning
    age_bins = [0, 25, 60, 120]
    age_labels = ['youth', 'working', 'senior']
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)

    # 3. Balance Log Transformation (Log1p handling negative values gracefully)
    df['balance_log'] = np.log1p(df['balance'].clip(lower=0))

    # 4. Historical Campaign Success Indicator
    df['was_previously_successful'] = (df['poutcome'] == 'success').astype(int)

    # 5. Campaign Call Fatigue Threshold
    df['excessive_calls'] = (df['campaign'] > 4).astype(int)

    # 6. Seasonal Month Tiering
    high_tier_months = ['mar', 'sep', 'oct', 'dec']
    low_tier_months = ['may', 'jul', 'aug']
    
    def classify_month_tier(m):
        m_str = str(m).lower()
        if m_str in high_tier_months:
            return 'high'
        elif m_str in low_tier_months:
            return 'low'
        return 'medium'

    df['month_tier'] = df['month'].apply(classify_month_tier)

    # 7. One-Hot Encoding Categorical Features
    categorical_cols = ['job', 'marital', 'education', 'contact', 'month', 'poutcome', 'age_group', 'month_tier']
    existing_cat_cols = [col for col in categorical_cols if col in df.columns]
    
    # Binary Mapping for Remaining Binary Features
    binary_cols = ['default', 'housing', 'loan']
    for col in binary_cols:
        if col in df.columns and pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.lower().map({'yes': 1, 'no': 0}).fillna(0).astype(int)
        elif col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].map({'yes': 1, 'no': 0}).fillna(0).astype(int)

    # Perform One-Hot Encoding
    df = pd.get_dummies(df, columns=existing_cat_cols, drop_first=True)
    
    logging.info(f"Feature engineering complete. Engineered dataset shape: {df.shape}")
    return df

if __name__ == "__main__":
    raw_path = r"D:\BankWise Term Deposit Propensity & Risk Scoring Engine\BankWise-Term-Deposit-Prediction\data\processed\banking_cleaned.csv"
    output_path = r"D:\BankWise Term Deposit Propensity & Risk Scoring Engine\BankWise-Term-Deposit-Prediction\data\processed\banking_engineered.csv"

    if not os.path.exists(raw_path):
        raw_path = "data/raw/banking_data.csv"
        
    logging.info(f"Loading data from {raw_path}")
    df = pd.read_csv(raw_path)
    
    # Ensure target 'y' is binary integer
    if 'y' in df.columns and df['y'].dtype == 'object':
        df['y'] = df['y'].map({'yes': 1, 'no': 0}).astype(int)

    engineered_df = engineer_features(df)
    
    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    engineered_df.to_csv(output_path, index=False)
    logging.info(f"Saved engineered dataset to {output_path}")