import os
import pandas as pd
import numpy as np
import joblib
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def generate_explainability_artifacts():
    # 1. Paths
    model_path = "models/champion_model.pkl"
    data_path = "data/processed/banking_engineered.csv"

    if not os.path.exists(data_path):
        data_path = r"D:\BankWise Term Deposit Propensity & Risk Scoring Engine\BankWise-Term-Deposit-Prediction\data\processed\banking_engineered.csv"
        model_path = r"D:\BankWise Term Deposit Propensity & Risk Scoring Engine\BankWise-Term-Deposit-Prediction\models\champion_model.pkl"

    logging.info(f"Loading champion model from {model_path}")
    model = joblib.load(model_path)

    logging.info(f"Loading engineered dataset from {data_path}")
    df = pd.read_csv(data_path)

    X = df.drop(columns=['y'])
    y = df['y']

    _, X_test, _, _ = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    # 2. Extract Feature Importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = X.columns
        
        fi_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        print("\n" + "="*50)
        print(" TOP 10 FEATURE IMPORTANCE DRIVERS ")
        print("="*50)
        print(fi_df.head(10).to_string(index=False))

        # Save feature importance CSV and Plot
        os.makedirs("reports/figures", exist_ok=True)
        fi_df.to_csv("reports/feature_importances.csv", index=False)

        plt.figure(figsize=(12, 6))
        plt.barh(fi_df['Feature'].head(10)[::-1], fi_df['Importance'].head(10)[::-1], color='navy')
        plt.xlabel('Importance Score')
        plt.title('Top 10 Global Feature Importance Drivers (Champion Model)')
        plt.tight_layout()
        plt.savefig("reports/figures/feature_importance_summary.png", dpi=300)
        plt.close()
        logging.info("Saved feature importance chart to 'reports/figures/feature_importance_summary.png'")

    # 3. SHAP Explanation Setup (Using SHAP library if available)
    try:
        import shap
        logging.info("Calculating SHAP values for sample test set...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test.iloc[:500])

        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test.iloc[:500], show=False)
        plt.tight_layout()
        plt.savefig("reports/figures/shap_summary_plot.png", dpi=300)
        plt.close()
        logging.info("Saved SHAP summary plot to 'reports/figures/shap_summary_plot.png'")
    except ImportError:
        logging.warning("SHAP package not installed. Skipping SHAP plot generation. Install via 'pip install shap'.")

if __name__ == "__main__":
    generate_explainability_artifacts()