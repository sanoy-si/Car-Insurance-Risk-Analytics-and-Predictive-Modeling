import pandas as pd

def prepare_claim_severity_data(df):
    df = df[df['TotalClaims'] > 0].copy()
    df.dropna(inplace=True)
    df['VehicleAge'] = 2025 - df['VehicleYear']
    categorical_cols = ['Province', 'CoverType', 'AutoMake']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df
