import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model
import tensorflow as tf
import geopandas as gpd

# Import the new post-processing function
from post_processing import calculate_weighted_coordinates

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

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

def create_sequences(features, targets, time_steps=10):
    """
    Converts time-series data into sequences for LSTM evaluation.
    """
    X_seq, y_seq, target_indices = [], [], []
    padded_features = np.vstack([np.zeros((time_steps-1, features.shape[1])), features])
    for i in range(len(targets)):
        X_seq.append(padded_features[i:i+time_steps])
        y_seq.append(targets[i])
        target_indices.append(i)
    return np.array(X_seq), np.array(y_seq), np.array(target_indices)

def recreate_grid_from_fixed_boundaries(df, grid_size_meters=10):
    """
    Recreates the fixed grid over the entire ITU Ayazağa campus.
    """
    lat_min, lat_max = 41.098692000, 41.110922000
    lon_min, lon_max = 29.014443000, 29.037912000
    lat_step = grid_size_meters / 111000
    mean_latitude_campus = (lat_min + lat_max) / 2
    lon_step = grid_size_meters / (111000 * np.cos(np.radians(mean_latitude_campus)))
    lat_bins = np.arange(lat_min, lat_max + lat_step, lat_step)
    lon_bins = np.arange(lon_min, lon_max + lon_step, lon_step)
    df['lat_cell'] = pd.cut(df['Latitude'], bins=lat_bins, labels=False, include_lowest=True)
    df['lon_cell'] = pd.cut(df['Longitude'], bins=lon_bins, labels=False, include_lowest=True)
    df['cell_id'] = df['lat_cell'].astype(str) + '_' + df['lon_cell'].astype(str)
    return df

def evaluate_models(data_path):
    """
    Loads trained models and evaluates their performance side-by-side using
    the probability-weighted coordinate calculation method.
    """
    # --- 1. Load Data and Saved Assets ---
    script_dir = os.path.dirname(os.path.realpath(__file__))
    models_dir = os.path.join(script_dir, "Models")
    visuals_dir = os.path.join(script_dir, "Visuals")
    os.makedirs(visuals_dir, exist_ok=True)
    
    try:
        df = pd.read_excel(os.path.join(script_dir, data_path))
        label_encoder = joblib.load(os.path.join(models_dir, 'label_encoder.joblib'))
        cell_centers = joblib.load(os.path.join(models_dir, 'cell_centers.joblib'))
        scaler = joblib.load(os.path.join(models_dir, 'feature_scaler.joblib'))
        feature_names = joblib.load(os.path.join(models_dir, 'feature_names.joblib'))
        print("✅ All necessary data and assets loaded.")
    except FileNotFoundError as e:
        print(f"❌ Error loading assets: {e}. Please run the training script first.")
        return

    # --- 2. Prepare Data ---
    df_gridded = recreate_grid_from_fixed_boundaries(df, grid_size_meters=10)
    known_cells = label_encoder.classes_
    df_filtered = df_gridded[df_gridded['cell_id'].isin(known_cells)].copy().reset_index(drop=True)
    
    if df_filtered.empty:
        print("❌ Error: No known cell IDs found in the dataset.")
        return
        
    y = df_filtered['cell_id']
    X = df_filtered[feature_names]
    X_scaled = scaler.transform(X)
    y_encoded = label_encoder.transform(y)
    
    # --- 3. Create consistent test sets ---
    _, X_test_tree, _, y_test_tree, _, test_indices_tree = train_test_split(
        X_scaled, y_encoded, df_filtered.index, test_size=0.2, random_state=42, stratify=y_encoded
    )
    TIME_STEPS = 10
    X_seq, y_seq, seq_indices = create_sequences(X_scaled, y_encoded, TIME_STEPS)
    _, X_test_lstm, _, y_test_lstm, _, test_indices_lstm_seq = train_test_split(
        X_seq, y_seq, seq_indices, test_size=0.2, random_state=42, stratify=y_seq
    )
    test_indices_lstm = df_filtered.index[test_indices_lstm_seq]

    # --- 4. Generate Predictions (Probabilities) ---
    predictions = {}
    
    # Tree-based models
    for model_name in ['XGBoost', 'RandomForest']:
        model_path = os.path.join(models_dir, f'{model_name}_model.joblib')
        if os.path.exists(model_path):
            print(f"\n--- Evaluating {model_name} ---")
            model = joblib.load(model_path)
            # Get probabilities instead of final predictions
            y_pred_proba = model.predict_proba(X_test_tree)
            predictions[model_name] = {
                'pred_proba': y_pred_proba, 
                'true': y_test_tree, 
                'indices': test_indices_tree,
            }

    # LSTM model
    lstm_path = os.path.join(models_dir, 'LSTM_model.keras')
    if os.path.exists(lstm_path):
        print("\n--- Evaluating LSTM ---")
        model = load_model(lstm_path)
        y_pred_proba = model.predict(X_test_lstm, verbose=0)
        predictions['LSTM'] = {
            'pred_proba': y_pred_proba, 
            'true': y_test_lstm, 
            'indices': test_indices_lstm,
        }

    # --- 5. Generate detailed reports and visualizations using weighted method ---
    results_data = []
    
    for name, data in predictions.items():
        print(f"\n{'='*50}")
        print(f"--- Detailed Evaluation Report for {name} ---")
        print(f"{'='*50}")
        
        y_pred_proba = data['pred_proba']
        y_true = data['true']
        eval_indices = data['indices']
        
        # --- Classification metrics (based on top-1 prediction) ---
        y_pred_top1 = np.argmax(y_pred_proba, axis=1)
        accuracy = accuracy_score(y_true, y_pred_top1)
        print(f"Top-1 Accuracy: {accuracy:.4f}\n")
        
        # --- Location Error Calculation (using weighted method) ---
        true_coords = df_filtered.loc[eval_indices][['Latitude', 'Longitude']].values
        
        # Use the post-processing function to get refined coordinates
        pred_coords_weighted = calculate_weighted_coordinates(y_pred_proba, label_encoder, cell_centers, top_n=3)
        
        # Filter out any NaN results if a cell_id was not in cell_centers
        valid_indices = ~np.isnan(pred_coords_weighted).any(axis=1)
        if np.any(valid_indices):
            distances = haversine_distance(
                true_coords[valid_indices, 0], true_coords[valid_indices, 1], 
                pred_coords_weighted[valid_indices, 0], pred_coords_weighted[valid_indices, 1]
            )
            mean_error = np.mean(distances)
            median_error = np.median(distances)
            percentile_90 = np.percentile(distances, 90)
            
            print(f"Location Error Statistics (Weighted):")
            print(f"  Mean Error: {mean_error:.2f} meters")
            print(f"  Median Error: {median_error:.2f} meters")
            print(f"  90th Percentile: {percentile_90:.2f} meters")
        else:
            mean_error, median_error, percentile_90 = [np.inf] * 3
            print("\nWarning: No valid predictions for error calculation")
            
        results_data.append({
            'Model': name, 
            'Accuracy (%)': accuracy * 100, 
            'Mean Error (m)': mean_error,
            'Median Error (m)': median_error,
            '90th Percentile (m)': percentile_90,
        })

    # --- 6. Final Summary and Visualizations ---
    if not results_data:
        print("\n❌ No models were evaluated. Exiting.")
        return
        
    results_df = pd.DataFrame(results_data).round(2)
    print("\n" + "="*70)
    print("FINAL PERFORMANCE SUMMARY (PROBABILITY WEIGHTED)")
    print("="*70)
    print(results_df.to_string(index=False))
    
    # Save results to CSV
    results_df.to_csv(os.path.join(visuals_dir, 'model_performance_results_weighted.csv'), index=False)
    print(f"\n✅ Weighted results saved to {os.path.join(visuals_dir, 'model_performance_results_weighted.csv')}")

    # Create comparison bar plot
    plt.figure(figsize=(14, 7))
    metrics_to_plot = ['Accuracy (%)', 'Mean Error (m)', 'Median Error (m)']
    results_melted = results_df.melt(id_vars='Model', value_vars=metrics_to_plot, var_name='Metric', value_name='Value')
    
    sns.barplot(data=results_melted, x='Metric', y='Value', hue='Model', palette='viridis')
    plt.title('Model Performance Comparison (Weighted Coordinates)')
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.savefig(os.path.join(visuals_dir, 'model_comparison_weighted.png'), dpi=300)
    plt.show()

if __name__ == '__main__':
    DATA_PATH = os.path.join("Data", "Merged_encoded_filled_filtered_pciCleaned_featureEngineered.xlsx")
    evaluate_models(DATA_PATH)
