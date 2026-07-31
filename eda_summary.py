import os
import pandas as pd

# Automatically locate project root folder (1 level above /src)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Point to your CSV file path (adjust filename/subfolder if needed)
data_path = os.path.join(BASE_DIR, "data", "processed", "banking_engineered.csv")

# Load dataset
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
else:
    # If using bank-full.csv raw file:
    raw_path = os.path.join(BASE_DIR, "data", "raw", "bank-full.csv")
   

# ---------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------
# Adjust path as needed for your project structure
df = pd.read_csv("data/processed/banking_engineered.csv")

# Quick sanity check
print(f"Dataset Shape: {df.shape}")
df.head()

# Binary encoding for target variable analysis
df["target_binary"] = df["y"].map({"no": 0, "yes": 1})


# =========================================================
# DEMOGRAPHIC & FINANCIAL PROFILES (Q1 - Q8)
# =========================================================

# Q1: Age Distribution
print("--- 1. Age Stats ---")
print(df["age"].describe())

plt.figure(figsize=(8, 4))
sns.histplot(df["age"], kde=True, bins=30, color="#38BDF8")
plt.title("Client Age Distribution")
plt.xlabel("Age")
plt.ylabel("Client Count")
plt.show()

# Q2: Job Type Variation
print("\n--- 2. Job Type Counts ---")
job_counts = df["job"].value_counts(normalize=True) * 100
print(job_counts.round(2))

plt.figure(figsize=(10, 4))
sns.countplot(
    data=df,
    y="job",
    order=df["job"].value_counts().index,
    palette="Blues_r",
    hue="job",
    legend=False,
)
plt.title("Client Distribution by Job Type")
plt.xlabel("Count")
plt.show()

# Q3: Marital Status
print("\n--- 3. Marital Status ---")
print(df["marital"].value_counts(normalize=True).round(4) * 100)

# Q4: Education Level
print("\n--- 4. Education Levels ---")
print(df["education"].value_counts(normalize=True).round(4) * 100)

# Q5: Credit Default Proportion
print("\n--- 5. Credit in Default ---")
default_pct = df["default"].value_counts(normalize=True) * 100
print(f"Default Breakdown (%):\n{default_pct.round(2)}")

# Q6: Yearly Balance Distribution
print("\n--- 6. Balance Summary ---")
print(df["balance"].describe())

# Extreme skew check
plt.figure(figsize=(8, 4))
sns.boxplot(x=df["balance"], color="#10B981")
plt.title("Average Yearly Balance (€)")
plt.show()

# Q7 & Q8: Housing and Personal Loans
print("\n--- 7 & 8. Loan Holdings ---")
print("Housing Loan (%):")
print((df["housing"].value_counts(normalize=True) * 100).round(2))
print("\nPersonal Loan (%):")
print((df["loan"].value_counts(normalize=True) * 100).round(2))


# =========================================================
# CAMPAIGN & CONTACT METRICS (Q9 - Q16)
# =========================================================

# Q9: Communication Types
print("\n--- 9. Contact Communication Type ---")
print(df["contact"].value_counts(normalize=True).round(4) * 100)

# Q10: Last Contact Day of Month
plt.figure(figsize=(9, 3))
sns.histplot(df["day"], bins=31, color="#818CF8")
plt.title("Distribution of Contact Day of Month")
plt.xlabel("Day of Month")
plt.show()

# Q11: Last Contact Month
month_order = [
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
]
print("\n--- 11. Contact Month Breakdown ---")
print(df["month"].value_counts()[month_order])

# Q12: Call Duration Distribution
print("\n--- 12. Duration Stats (Seconds) ---")
print(df["duration"].describe())

# Q13: Contacts During Current Campaign
print("\n--- 13. Campaign Call Counts ---")
print(df["campaign"].describe(percentiles=[0.5, 0.75, 0.90, 0.95]))

# Q14: Days Since Previous Contact (pdays)
never_contacted = (df["pdays"] == -1).sum()
print(
    f"\n--- 14. Days Since Last Contact ---\nNever Contacted Ratio: {never_contacted / len(df):.2%}"
)

# Q15: Number of Contacts Before Current Campaign
print("\n--- 15. Previous Contact Counts ---")
print(df["previous"].value_counts().head(10))

# Q16: Previous Campaign Outcome
print("\n--- 16. Previous Campaign Outcomes ---")
print(df["poutcome"].value_counts(normalize=True).round(4) * 100)


# =========================================================
# TARGET VARIABLE & CORRELATIONS (Q17 - Q18)
# =========================================================

# Q17: Target Variable Distribution (y)
print("\n--- 17. Subscription Target Breakdown ---")
target_counts = df["y"].value_counts()
target_props = df["y"].value_counts(normalize=True)
print(
    pd.DataFrame({"Count": target_counts, "Percentage": target_props * 100}).round(
        2
    )
)

plt.figure(figsize=(5, 4))
sns.countplot(data=df, x="y", palette=["#EF4444", "#10B981"], hue="y")
plt.title("Term Deposit Subscriptions (Target variable 'y')")
plt.show()

# Q18: Feature Relationships & Conversion Drivers
print("\n--- 18. Correlation Analysis ---")

# Numeric correlations with target
numeric_df = df.select_dtypes(include=[np.number])
correlations = numeric_df.corr()["target_binary"].sort_values(ascending=False)
print("Numeric Feature Correlations with Target:")
print(correlations.round(4))

# Key Categorical Drivers
print("\nConversion Rate by Previous Campaign Outcome:")
poutcome_conv = (
    df.groupby("poutcome")["target_binary"].mean().sort_values(ascending=False)
    * 100
)
print(poutcome_conv.round(2))

print("\nConversion Rate by Job Type:")
job_conv = (
    df.groupby("job")["target_binary"].mean().sort_values(ascending=False)
    * 100
)
print(job_conv.round(2))

# Visualizing Top Drivers
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.barplot(
    x=poutcome_conv.index,
    y=poutcome_conv.values,
    ax=axes[0],
    palette="Blues_r",
    hue=poutcome_conv.index,
    legend=False,
)
axes[0].set_title("Conversion Rate by Previous Outcome (%)")
axes[0].set_ylabel("Conversion %")

sns.boxplot(
    data=df, x="y", y="duration", ax=axes[1], palette=["#EF4444", "#10B981"]
)
axes[1].set_title("Call Duration vs Subscription")
axes[1].set_yscale("log")  # Using log scale due to heavy skewness

plt.tight_layout()
plt.show()