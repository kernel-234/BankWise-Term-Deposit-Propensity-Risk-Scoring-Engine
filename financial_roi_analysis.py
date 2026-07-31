import os
import pandas as pd
import numpy as np
import joblib
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def perform_financial_roi_analysis():
    # 1. Load Model & Test Data
    model_path = "models/champion_model.pkl"
    data_path = "data/processed/banking_engineered.csv"

    if not os.path.exists(data_path):
        data_path = r"D:\BankWise Term Deposit Propensity & Risk Scoring Engine\BankWise-Term-Deposit-Prediction\data\processed\banking_engineered.csv"
        model_path = r"D:\BankWise Term Deposit Propensity & Risk Scoring Engine\BankWise-Term-Deposit-Prediction\models\champion_model.pkl"

    logging.info(f"Loading champion model from {model_path}")
    model = joblib.load(model_path)

    df = pd.read_csv(data_path)
    X = df.drop(columns=['y'])
    y = df['y']

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    # 2. Financial Financial Parameters
    CALL_COST = 5.0      # €5 per call
    DEPOSIT_VALUE = 150.0 # €150 revenue per converted term deposit

    # 3. Predict Probabilities
    probs = model.predict_proba(X_test)[:, 1]

    # Scenario A: Traditional Approach (Call Everyone in Test Set)
    total_clients = len(y_test)
    actual_conversions = y_test.sum()

    cost_traditional = total_clients * CALL_COST
    revenue_traditional = actual_conversions * DEPOSIT_VALUE
    profit_traditional = revenue_traditional - cost_traditional
    roi_traditional = (profit_traditional / cost_traditional) * 100

    # Scenario B: ML-Driven Targeted Approach (Apply Champion Threshold = 0.7687)
    optimal_threshold = 0.7687
    preds_ml = (probs >= optimal_threshold).astype(int)

    calls_made_ml = preds_ml.sum()
    tp_ml = ((preds_ml == 1) & (y_test == 1)).sum() # True Positives captured

    cost_ml = calls_made_ml * CALL_COST
    revenue_ml = tp_ml * DEPOSIT_VALUE
    profit_ml = revenue_ml - cost_ml
    roi_ml = (profit_ml / cost_ml) * 100 if cost_ml > 0 else 0

    # Summary Metrics
    cost_savings = cost_traditional - cost_ml
    call_reduction_pct = ((total_clients - calls_made_ml) / total_clients) * 100

    print("\n" + "="*70)
    print(" FINANCIAL IMPACT & CAMPAIGN ROI ANALYSIS ")
    print("="*70)
    print(f"Total Test Set Prospects Evaluated   : {total_clients:,}")
    print("-" * 70)
    print("TRADITIONAL APPROACH (Call Everyone):")
    print(f"  • Calls Placed                     : {total_clients:,}")
    print(f"  • Total Campaign Cost              : €{cost_traditional:,.2f}")
    print(f"  • Total Revenue Generated          : €{revenue_traditional:,.2f}")
    print(f"  • Net Campaign Profit              : €{profit_traditional:,.2f}")
    print(f"  • Campaign ROI                     : {roi_traditional:.2f}%")
    print("-" * 70)
    print("BANKWISE ML-POWERED APPROACH (Threshold = 0.7687):")
    print(f"  • Calls Placed (Targeted Leads)    : {calls_made_ml:,}")
    print(f"  • Total Campaign Cost              : €{cost_ml:,.2f}")
    print(f"  • Total Revenue Generated          : €{revenue_ml:,.2f}")
    print(f"  • Net Campaign Profit              : €{profit_ml:,.2f}")
    print(f"  • Campaign ROI                     : {roi_ml:.2f}%")
    print("-" * 70)
    print("EXECUTIVE BUSINESS SAVINGS:")
    print(f"  • Outbound Call Volume Reduction   : {call_reduction_pct:.2f}%")
    print(f"  • Total Campaign Expense Saved     : €{cost_savings:,.2f}")
    print("="*70)

    # Save Business Report
    roi_data = {
        'Metric': ['Calls Placed', 'Campaign Cost (€)', 'Revenue (€)', 'Net Profit (€)', 'ROI (%)'],
        'Traditional_Strategy': [total_clients, cost_traditional, revenue_traditional, profit_traditional, roi_traditional],
        'BankWise_ML_Strategy': [calls_made_ml, cost_ml, revenue_ml, profit_ml, roi_ml]
    }
    pd.DataFrame(roi_data).to_csv("reports/financial_roi_summary.csv", index=False)
    logging.info("Saved ROI summary table to 'reports/financial_roi_summary.csv'")

if __name__ == "__main__":
    perform_financial_roi_analysis()