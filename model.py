import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import joblib
import time

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the earth.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r * 1000 # in meters

def tune_and_compare_models_geometric(featured_data_path):
    """
    Loads data with geometric features, tunes, trains, and compares RF and XGBoost models.
    """
    # --- 1. Load Data ---
    try:
        df = pd.read_excel(featured_data_path, index_col='Time')
        print("✅ Data with geometric features loaded successfully.")
    except FileNotFoundError:
        print(f"❌ Error: The file was not found at {featured_data_path}")
        print("Please ensure the geometric feature engineering script has been run successfully.")
        return

    # --- 2. Define Features and Target ---
    y = df[['Latitude', 'Longitude']]
    # Use all numeric columns as features, which now includes the distance columns
    X = df.drop(columns=['Latitude', 'Longitude']).select_dtypes(include=np.number)
    print(f"   - Using {X.shape[1]} features, including geometric distances.")

    # --- 3. Time-Based Data Split ---
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    print(f"Data split into {len(X_train)} training and {len(X_test)} testing samples.")

    # --- 4. Hyperparameter Tuning Setup ---
    # Same grids as before, but the models will now see the new distance features
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_search = RandomizedSearchCV(estimator=rf, param_distributions=rf_param_grid, n_iter=10, cv=3, verbose=2, random_state=42, n_jobs=-1)

    xgb_param_grid = {
        'n_estimators': [100, 300, 500],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }
    xgbr = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
    xgb_search = RandomizedSearchCV(estimator=xgbr, param_distributions=xgb_param_grid, n_iter=10, cv=3, verbose=2, random_state=42, n_jobs=-1)

    # --- 5. Train and Evaluate Models ---
    models = {'RandomForest': rf_search, 'XGBoost': xgb_search}
    results = {}

    for name, search_cv in models.items():
        print(f"\n--- Tuning {name} with Geometric Features ---")
        start_time = time.time()
        search_cv.fit(X_train, y_train)
        end_time = time.time()
        
        print(f"Best parameters for {name}: {search_cv.best_params_}")
        
        best_model = search_cv.best_estimator_
        y_pred = best_model.predict(X_test)
        
        distances = haversine_distance(y_test['Latitude'], y_test['Longitude'], y_pred[:, 0], y_pred[:, 1])
        mean_error = np.mean(distances)
        
        results[name] = {'model': best_model, 'mean_error': mean_error, 'predictions': y_pred}
        print(f"✅ {name} Mean Positioning Error: {mean_error:.2f} meters")
        print(f"Tuning and training took {(end_time - start_time):.2f} seconds.")

    # --- 6. Compare and Select Best Model ---
    best_model_name = min(results, key=lambda k: results[k]['mean_error'])
    best_model_info = results[best_model_name]
    
    print(f"\n🏆 Best performing model with geometric features: {best_model_name} with {best_model_info['mean_error']:.2f}m error.")

    # --- 7. Visualize Best Model's Predictions ---
    print("📊 Visualizing best model's predictions...")
    visuals_dir = "Visuals"
    os.makedirs(visuals_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 12))
    plt.scatter(y_test['Longitude'], y_test['Latitude'], color='blue', label='Actual Path', s=15, alpha=0.6)
    plt.scatter(best_model_info['predictions'][:, 1], best_model_info['predictions'][:, 0], color='green', label=f'Predicted Path ({best_model_name})', s=15, alpha=0.6)
    plt.title(f'Actual vs. Predicted Location (Geometric Features - {best_model_name})')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(visuals_dir, f'best_model_predictions_geometric_{best_model_name}.png'))
    plt.show()

    # --- 8. Save the Best Model ---
    print("💾 Saving the best model...")
    models_dir = "Models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'best_positioning_model_geometric.joblib')
    
    joblib.dump(best_model_info['model'], model_path)
    print(f"   - Best model ({best_model_name}) successfully saved to {model_path}")

# --- Execution ---
if __name__ == '__main__':
    # Use the new data file with geometric features
    FEATURED_DATA_PATH = os.path.join("Data", "featured_data_geometric.xlsx")
    tune_and_compare_models_geometric(FEATURED_DATA_PATH)
