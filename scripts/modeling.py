from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

def train_models(X_train, y_train):
    lr = LinearRegression()
    rf = RandomForestRegressor(random_state=42, n_estimators=100)
    xgb_model = xgb.XGBRegressor(random_state=42, n_estimators=100, objective='reg:squarederror')

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    return lr, rf, xgb_model
