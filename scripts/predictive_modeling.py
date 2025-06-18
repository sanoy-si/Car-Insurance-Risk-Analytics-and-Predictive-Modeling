import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_preparation import prepare_claim_severity_data
from src.modeling import train_models
from src.interpretability import explain_with_shap

# Load data
df = pd.read_csv('data/processed/clean_data.csv')

# Prepare data
df_prepared = prepare_claim_severity_data(df)
X = df_prepared.drop(columns=['TotalClaims', 'PolicyID', 'TransactionMonth'])
y = df_prepared['TotalClaims']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
lr, rf, xgb_model = train_models(X_train, y_train)

# Explain with SHAP
explain_with_shap(xgb_model, X_train, X_test)
