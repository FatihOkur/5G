import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import VarianceThreshold 
import joblib
import time

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance in meters between two points on the earth.
    """
    lat1, lon1, lat2, lon2 = map(np.asarray, [lat1, lon1, lat2, lon2])
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 
    return c * r * 1000

def tune_and_evaluate_advanced_models(featured_data_path):
    """
    Loads data, cleans it, tunes models, and performs error diagnostics.
    """
    # --- 1. Load Data ---
    try:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        full_path = os.path.join(script_dir, featured_data_path)
        df = pd.read_excel(full_path, index_col='Time')
        print(f"✅ Data with advanced features loaded successfully from {full_path}")
    except FileNotFoundError:
        print(f"❌ Error: The file was not found at {full_path}")
        return

    # --- 2. Define Features and Target ---
    y = df[['Latitude', 'Longitude']]
    X = df.select_dtypes(include=np.number).drop(columns=['Latitude', 'Longitude'], errors='ignore')
    
    # --- 3. Additional Cleaning: Remove Low-Variance Features ---
    initial_feature_count = X.shape[1]
    print(f"   - Original number of features: {initial_feature_count}")
    selector = VarianceThreshold(threshold=0.01)
    X_high_variance = selector.fit_transform(X)
    kept_columns = X.columns[selector.get_support(indices=True)]
    X = pd.DataFrame(X_high_variance, index=X.index, columns=kept_columns)
    final_feature_count = X.shape[1]
    print(f"   - Number of features after removing low variance columns: {final_feature_count}")
    print(f"   - Removed {initial_feature_count - final_feature_count} quasi-constant features.") # <-- Fixed the print statement

    # --- 4. Time-Based Data Split ---
    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    print(f"Data split into {len(X_train)} training and {len(X_test)} testing samples.")

    # --- 5. Hyperparameter Tuning Setup ---
    n_tuning_iterations = 20
    rf_param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, 30, None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_search = RandomizedSearchCV(estimator=rf, param_distributions=rf_param_grid, n_iter=n_tuning_iterations, cv=3, verbose=1, random_state=42, n_jobs=-1)

    xgb_param_grid = {'n_estimators': [100, 300, 500], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7], 'subsample': [0.7, 0.8, 0.9], 'colsample_bytree': [0.7, 0.8, 0.9]}
    xgbr = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
    xgb_search = RandomizedSearchCV(estimator=xgbr, param_distributions=xgb_param_grid, n_iter=n_tuning_iterations, cv=3, verbose=1, random_state=42, n_jobs=-1)

    # --- 6. Train and Evaluate Models ---
    models = {'RandomForest': rf_search, 'XGBoost': xgb_search}
    results = {}

    for name, search_cv in models.items():
        print(f"\n--- Tuning {name} with Final Cleaned Features ---")
        start_time = time.time()
        search_cv.fit(X_train, y_train)
        end_time = time.time()
        
        print(f"Best parameters for {name}: {search_cv.best_params_}")
        best_model = search_cv.best_estimator_
        y_pred = best_model.predict(X_test)
        
        distances = haversine_distance(y_test['Latitude'], y_test['Longitude'], y_pred[:, 0], y_pred[:, 1])
        mean_error = np.mean(distances)
        
        results[name] = {'model': best_model, 'mean_error': mean_error, 'predictions': y_pred, 'errors': distances}
        print(f"✅ {name} Mean Positioning Error: {mean_error:.2f} meters")
        print(f"Tuning and training took {(end_time - start_time):.2f} seconds.")

    # --- 7. Compare and Select Best Model ---
    best_model_name = min(results, key=lambda k: results[k]['mean_error'])
    best_model_info = results[best_model_name]
    print(f"\n🏆 Best performing model with final features: {best_model_name} with {best_model_info['mean_error']:.2f}m error.")

    # --- 8. Save the Best Model ---
    print("💾 Saving the best model...")
    script_dir = os.path.dirname(os.path.realpath(__file__))
    models_dir = os.path.join(script_dir, "Models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'best_positioning_model_final.joblib')
    joblib.dump(best_model_info['model'], model_path)
    print(f"   - Best model ({best_model_name}) successfully saved to {model_path}")

    # --- 9. NEW: Diagnostics and Visualization ---
    print("\n--- ախ DIAGNOSTICS ---")
    visuals_dir = os.path.join(script_dir, "Visuals")
    os.makedirs(visuals_dir, exist_ok=True)

    # Plot Feature Importance for the best model
    if best_model_name == 'XGBoost':
        best_model = best_model_info['model']
        fig, ax = plt.subplots(figsize=(12, 8))
        xgb.plot_importance(best_model, max_num_features=20, height=0.5, ax=ax, title=f'Feature Importance ({best_model_name})')
        plt.tight_layout()
        plt.savefig(os.path.join(visuals_dir, 'feature_importance.png'))
        print("📊 Feature importance plot saved to Visuals/feature_importance.png")
        plt.show()

    # Analyze the worst predictions
    errors_df = pd.DataFrame({
        'error': best_model_info['errors'],
        'pred_lat': best_model_info['predictions'][:, 0],
        'pred_lon': best_model_info['predictions'][:, 1]
    }, index=y_test.index)
    
    worst_predictions = errors_df.sort_values('error', ascending=False).head(10)
    print("\n--- Top 10 Worst Predictions ---")
    # Join with original features to see the context
    worst_predictions_with_features = X_test.join(worst_predictions, how='inner')
    print(worst_predictions_with_features)

    # Visualize predictions with high-error points highlighted
    plt.figure(figsize=(12, 12))
    plt.scatter(y_test['Longitude'], y_test['Latitude'], color='blue', label='Actual Path', s=15, alpha=0.4)
    plt.scatter(best_model_info['predictions'][:, 1], best_model_info['predictions'][:, 0], color='green', label=f'Predicted Path ({best_model_name})', s=15, alpha=0.4)
    # Highlight the worst prediction points in red
    plt.scatter(worst_predictions_with_features['pred_lon'], worst_predictions_with_features['pred_lat'], color='red', s=50, edgecolor='black', label='Worst Predictions')
    plt.title('Actual vs. Predicted Location with High-Error Points')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(visuals_dir, f'predictions_with_error_analysis.png'))
    print("📊 Prediction plot with error analysis saved to Visuals/predictions_with_error_analysis.png")
    plt.show()


# --- Execution ---
if __name__ == '__main__':
    ADVANCED_FEATURED_DATA_PATH = os.path.join("Data", "featured_data_final_v3.xlsx")
    tune_and_evaluate_advanced_models(ADVANCED_FEATURED_DATA_PATH)
