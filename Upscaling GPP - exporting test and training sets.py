import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

# Load data from Excel
file_path = r'J:\Final\GPP_NDVI_DAY.xlsx'  # Replace with your Excel file path
data = pd.read_excel(file_path)

# Extract features, target variable, and metadata columns to retain
X = data[['Feature2', 'Feature6', 'Feature8', 'Feature10']].copy()  # Features for modeling
y = data['Target'].copy()  # Target variable
metadata = data[['station', 'time']].copy()  # Columns to keep but not used in modeling

# Handle missing values
X.fillna(X.mean(), inplace=True)
y.fillna(y.mean(), inplace=True)

# Split dataset - split features, target, and metadata simultaneously
X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
    X, y, metadata, test_size=0.2, random_state=42)

# Standardize features - fit scaler only on training set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit only on training set
X_test_scaled = scaler.transform(X_test)  # Transform test set using training scaler

# Create Random Forest model
model = RandomForestRegressor(random_state=42)

# Define hyperparameter grid for GridSearch
param_grid = {
    'n_estimators': [300],
    'max_depth': [20],
    'min_samples_split': [6],
    'min_samples_leaf': [3],
    'max_features': [3]
}

# Perform Grid Search with cross-validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, scoring='r2', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Predict on training and test sets using the best model
y_pred_train_best = best_model.predict(X_train_scaled)
y_pred_test_best = best_model.predict(X_test_scaled)

# Combine train and test sets to compute overall R² and RMSE
y_all_best = np.concatenate([y_train, y_test])
X_all_best = np.concatenate([X_train_scaled, X_test_scaled], axis=0)
y_pred_all_best = best_model.predict(X_all_best)

# Compute R² scores
r2_train_best = r2_score(y_train, y_pred_train_best)
r2_test_best = r2_score(y_test, y_pred_test_best)
r2_all_best = r2_score(y_all_best, y_pred_all_best)

# Compute RMSE scores
rmse_train_best = np.sqrt(mean_squared_error(y_train, y_pred_train_best))
rmse_test_best = np.sqrt(mean_squared_error(y_test, y_pred_test_best))
rmse_all_best = np.sqrt(mean_squared_error(y_all_best, y_pred_all_best))

# Print best hyperparameters and performance metrics
print(f'Best hyperparameters: {grid_search.best_params_}')
print(f'Best model R²_train: {r2_train_best:.4f}, RMSE_train: {rmse_train_best:.4f}')
print(f'Best model R²_test: {r2_test_best:.4f}, RMSE_test: {rmse_test_best:.4f}')
print(f'Best model R²_all: {r2_all_best:.4f}, RMSE_all: {rmse_all_best:.4f}')

# Save the best model and scaler to files
model_file_path = r'F:\Thesis\TPMFD\Model_Accuracy\GPP\best_random_forest_model_GPP.pkl'
scaler_file_path = r'F:\Thesis\TPMFD\Model_Accuracy\GPP\standard_scaler_GPP.pkl'
joblib.dump(best_model, model_file_path)
joblib.dump(scaler, scaler_file_path)
print(f'Best model saved to {model_file_path}')
print(f'StandardScaler saved to {scaler_file_path}')

# Load the saved model and scaler (for demonstration or reuse)
best_model = joblib.load(model_file_path)
scaler = joblib.load(scaler_file_path)

# Create DataFrame for training set results - include station and time
result_train_df = pd.DataFrame({
    'station': meta_train['station'].values,
    'time': meta_train['time'].values,
    'True_GPP_train': y_train.values,
    'Predicted_GPP_train': y_pred_train_best
})

# Create DataFrame for test set results - include station and time
result_test_df = pd.DataFrame({
    'station': meta_test['station'].values,
    'time': meta_test['time'].values,
    'True_GPP_test': y_test.values,
    'Predicted_GPP_test': y_pred_test_best
})

# Export training results to Excel
output_train_file = r'F:\Thesis\TPMFD\Model_Accuracy\GPP\predicted_GPP_train.xlsx'
result_train_df.to_excel(output_train_file, index=False)
print(f'Training set prediction completed. Results saved to {output_train_file}')

# Export test results to Excel
output_test_file = r'F:\Thesis\TPMFD\Model_Accuracy\GPP\predicted_GPP_test.xlsx'
result_test_df.to_excel(output_test_file, index=False)
print(f'Test set prediction completed. Results saved to {output_test_file}')

# Load new data for prediction
new_file_path = r'J:\Final\DAY.xlsx'  # Replace with your new Excel file path
new_data = pd.read_excel(new_file_path)

# Extract feature columns
X_new = new_data[['Feature2', 'Feature6', 'Feature8', 'Feature10']].copy()

# Handle missing values in new data
X_new.fillna(X_new.mean(), inplace=True)

# Standardize new features using the trained scaler
X_scaled_new = scaler.transform(X_new)

# Make predictions on new data
y_pred_new = best_model.predict(X_scaled_new)

# Create result DataFrame - include station and time if available
if all(col in new_data.columns for col in ['station', 'time']):
    result_df = pd.DataFrame({
        'station': new_data['station'].values,
        'time': new_data['time'].values,
        'GPP_DT': y_pred_new
    })
else:
    result_df = pd.DataFrame({
        'GPP_DT': y_pred_new
    })

# Export predictions to Excel
output_file = r'F:\Thesis\TPMFD\Model_Accuracy\GPP\predicted_GPP_DT.xlsx'
result_df.to_excel(output_file, index=False)

# Print feature importances
feature_importances = best_model.feature_importances_
print(dict(zip(X.columns, feature_importances)))
print(f'Prediction completed. Results saved to {output_file}')