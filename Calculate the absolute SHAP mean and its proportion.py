import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import shap


# ========================== Data Processing and Model Training Function ==========================
def train_and_evaluate_model(file_path, feature_cols, target_col, model_name, param_grid):
    # Load data
    data = pd.read_excel(file_path)
    X = data[feature_cols].copy()
    y = data[target_col].copy()

    # Handle missing values
    X.fillna(X.mean(), inplace=True)
    y.fillna(y.mean(), inplace=True)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize training set
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Apply the same scaler to the test set
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = RandomForestRegressor(random_state=42)

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_

    # Evaluate model
    y_pred_test_best = best_model.predict(X_test_scaled)
    r2_test_best = r2_score(y_test, y_pred_test_best)
    rmse_test_best = np.sqrt(mean_squared_error(y_test, y_pred_test_best))

    print(f'==== {model_name} ====')
    print(f'Best hyperparameters: {grid_search.best_params_}')
    print(f'Best model R²_test: {r2_test_best:.4f}, RMSE_test: {rmse_test_best:.4f}')

    feature_importances = best_model.feature_importances_
    print(dict(zip(feature_cols, feature_importances)))

    # Save model
    model_file_path = rf'F:\TPMFD\best_random_forest_model_{model_name}.pkl'
    joblib.dump(best_model, model_file_path)
    print(f'Best model saved to {model_file_path}')

    return X_test_scaled, best_model, feature_cols, scaler


# ========================== Main Program ==========================
# File paths and feature column definitions
file_path_gpp = r'F:\GPP_NDVI_DAY.xlsx'
file_path_er = r'F:\ER_NDVI_DAY.xlsx'

# Updated feature columns (removed Feature7 and Feature5)
feature_cols_gpp = ['特征2', '特征6', '特征8', '特征10']  # Removed Feature7
feature_cols_er = ['特征3', '特征6', '特征8', '特征11']  # Removed Feature5

target_col = '目标'

# Parameter grid for GPP
param_grid_gpp = {
    'n_estimators': [300],
    'max_depth': [20],
    'min_samples_split': [6],
    'min_samples_leaf': [3],
    'max_features': [3]
}

# Parameter grid for ER
param_grid_er = {
    'n_estimators': [500],
    'max_depth': [20],
    'min_samples_split': [6],
    'min_samples_leaf': [3],
    'max_features': [4]
}

# Train and evaluate GPP model
X_test_gpp, best_model_gpp, feature_cols_gpp, scaler_gpp = train_and_evaluate_model(
    file_path_gpp, feature_cols_gpp, target_col, 'GPP', param_grid_gpp
)

# Train and evaluate ER model
X_test_er, best_model_er, feature_cols_er, scaler_er = train_and_evaluate_model(
    file_path_er, feature_cols_er, target_col, 'ER', param_grid_er
)

# Feature name mapping (with removed features excluded)
feature_mapping_gpp = {
    '特征2': 'DR',
    '特征6': 'NDVI',
    '特征8': 'LAI',
    '特征10': 'LAI_MAX'
}

feature_mapping_er = {
    '特征3': 'Ta',
    '特征6': 'NDVI',
    '特征8': 'LAI',
    '特征11': 'LAI_RC'
}

# Replace feature names with mapped labels
feature_cols_gpp_labels = [feature_mapping_gpp[col] for col in feature_cols_gpp]
feature_cols_er_labels = [feature_mapping_er[col] for col in feature_cols_er]

# Compute SHAP values
explainer_gpp = shap.TreeExplainer(best_model_gpp)
shap_values_gpp = explainer_gpp.shap_values(X_test_gpp)

explainer_er = shap.TreeExplainer(best_model_er)
shap_values_er = explainer_er.shap_values(X_test_er)

# Compute mean absolute SHAP values for each feature
mean_shap_gpp = np.mean(np.abs(shap_values_gpp), axis=0)
mean_shap_er = np.mean(np.abs(shap_values_er), axis=0)

# Compute the proportion of each feature's mean SHAP value relative to the total
total_shap_gpp = np.sum(mean_shap_gpp)
total_shap_er = np.sum(mean_shap_er)

shap_ratio_gpp = mean_shap_gpp / total_shap_gpp
shap_ratio_er = mean_shap_er / total_shap_er

# Save results as DataFrames
results_gpp = pd.DataFrame({
    'Feature': feature_cols_gpp_labels,
    'Mean SHAP Value': mean_shap_gpp,
    'SHAP Ratio': shap_ratio_gpp
}).round(5)  # Round to 5 decimal places

results_er = pd.DataFrame({
    'Feature': feature_cols_er_labels,
    'Mean SHAP Value': mean_shap_er,
    'SHAP Ratio': shap_ratio_er
}).round(5)  # Round to 5 decimal places

# Export to Excel files
results_gpp.to_excel(r'F:\Thesis\TPMFD\GPP_SHAP_Result.xlsx', index=False)
results_er.to_excel(r'F:\Thesis\TPMFD\Reco_SHAP_Result.xlsx', index=False)

print("GPP SHAP results saved to F:\\Thesis\\TPMFD\\GPP_SHAP_Result.xlsx")
print("Reco SHAP results saved to F:\\Thesis\\TPMFD\\Reco_SHAP_Result.xlsx")