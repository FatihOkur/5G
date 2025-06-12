import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

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
    Enhanced with padding for consistency.
    """
    X_seq, y_seq, target_indices = [], [], []
    
    # Pad the beginning with zeros
    padded_features = np.vstack([np.zeros((time_steps-1, features.shape[1])), features])
    
    for i in range(len(targets)):
        X_seq.append(padded_features[i:i+time_steps])
        y_seq.append(targets[i])
        target_indices.append(i)
    
    return np.array(X_seq), np.array(y_seq), np.array(target_indices)

def recreate_grid_from_fixed_boundaries(df, grid_size_meters=10):
    """
    Recreates the fixed grid over the entire ITU Ayazağa campus to assign 
    cell_id to the evaluation data.
    """
    # Fixed ITU Campus Boundaries
    lat_min = 41.098692000
    lat_max = 41.110922000
    lon_min = 29.014443000
    lon_max = 29.037912000

    lat_step = grid_size_meters / 111000
    mean_latitude_campus = (lat_min + lat_max) / 2
    lon_step = grid_size_meters / (111000 * np.cos(np.radians(mean_latitude_campus)))

    lat_bins = np.arange(lat_min, lat_max + lat_step, lat_step)
    lon_bins = np.arange(lon_min, lon_max + lon_step, lon_step)

    df['lat_cell'] = pd.cut(df['Latitude'], bins=lat_bins, labels=False, include_lowest=True)
    df['lon_cell'] = pd.cut(df['Longitude'], bins=lon_bins, labels=False, include_lowest=True)
    df['cell_id'] = df['lat_cell'].astype(str) + '_' + df['lon_cell'].astype(str)
    return df

def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    """
    Create and save a confusion matrix plot.
    """
    # Limit to top 20 classes for visualization
    unique_classes = np.unique(np.concatenate((y_true, y_pred)))
    if len(unique_classes) > 20:
        # Get the most frequent classes
        class_counts = pd.Series(y_true).value_counts()
        top_classes = class_counts.head(20).index
        mask = np.isin(y_true, top_classes) & np.isin(y_pred, top_classes)
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]
    else:
        y_true_filtered = y_true
        y_pred_filtered = y_pred
    
    cm = confusion_matrix(y_true_filtered, y_pred_filtered)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name} (Top Classes)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate_models(data_path):
    """
    Loads trained models and evaluates their performance side-by-side.
    """
    # Load Data and Saved Assets
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
        print(f"❌ Error loading assets: {e}")
        print("Please run classification_model.py first to train and save models.")
        return

    # Prepare Data
    df_gridded = recreate_grid_from_fixed_boundaries(df, grid_size_meters=10)
    print("✅ Fixed ITU grid system recreated for evaluation.")
    
    # Filter for known cells
    known_cells = label_encoder.classes_
    df_filtered = df_gridded[df_gridded['cell_id'].isin(known_cells)].copy().reset_index(drop=True)
    
    if df_filtered.empty:
        print("❌ Error: No known cell IDs found in the dataset.")
        return
        
    print(f"✅ Found {len(df_filtered)} rows with known cell IDs for evaluation.")

    # Prepare features
    y = df_filtered['cell_id']
    X = df_filtered.select_dtypes(include=np.number).drop(columns=['Latitude', 'Longitude', 'lat_cell', 'lon_cell'])
    
    # Ensure feature consistency
    X = X[feature_names]
    
    X_scaled = scaler.transform(X)
    y_encoded = label_encoder.transform(y)
    
    # Filter classes with sufficient samples
    y_series = pd.Series(y_encoded)
    class_counts = y_series.value_counts()
    classes_to_keep = class_counts[class_counts >= 5].index
    is_class_large_enough = y_series.isin(classes_to_keep)
    
    df_filtered = df_filtered[is_class_large_enough].reset_index(drop=True)
    X_scaled = X_scaled[is_class_large_enough.values]
    y_encoded = y_encoded[is_class_large_enough.values]
    
    print(f"✅ Filtered out small classes. Kept {len(df_filtered)} rows for splitting.")

    # Create test sets
    _, X_test_tree, _, y_test_tree, _, test_indices_tree = train_test_split(
        X_scaled, y_encoded, df_filtered.index, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Prepare LSTM data
    TIME_STEPS = 10
    X_seq, y_seq, seq_indices = create_sequences(X_scaled, y_encoded, TIME_STEPS)
    _, X_test_lstm, _, y_test_lstm, _, test_indices_lstm_seq = train_test_split(
        X_seq, y_seq, seq_indices, test_size=0.2, random_state=42, stratify=y_seq
    )
    test_indices_lstm = df_filtered.index[test_indices_lstm_seq]

    # Generate Predictions
    predictions = {}
    
    # Tree-based models
    for model_name in ['XGBoost', 'RandomForest']:
        model_path = os.path.join(models_dir, f'{model_name}_model.joblib')
        if os.path.exists(model_path):
            print(f"\n--- Evaluating {model_name} ---")
            model = joblib.load(model_path)
            start_time = time.time()
            y_pred = model.predict(X_test_tree)
            pred_time = time.time() - start_time
            predictions[model_name] = {
                'pred': y_pred, 
                'true': y_test_tree, 
                'indices': test_indices_tree,
                'time': pred_time
            }
            print(f"Prediction time: {pred_time:.3f} seconds")

    # LSTM model
    lstm_path = os.path.join(models_dir, 'LSTM_model.keras')
    if os.path.exists(lstm_path):
        print("\n--- Evaluating LSTM ---")
        model = load_model(lstm_path)
        start_time = time.time()
        y_pred_proba = model.predict(X_test_lstm, verbose=0)
        y_pred_lstm = np.argmax(y_pred_proba, axis=1)
        pred_time = time.time() - start_time
        predictions['LSTM'] = {
            'pred': y_pred_lstm, 
            'true': y_test_lstm, 
            'indices': test_indices_lstm,
            'time': pred_time
        }
        print(f"Prediction time: {pred_time:.3f} seconds")

    # Ensemble model (Weighted Voting)
    from collections import defaultdict

    available_models = [name for name in ['XGBoost', 'RandomForest', 'LSTM'] if name in predictions]

    if len(available_models) >= 2:
        print("\n--- Evaluating Ensemble (Weighted Voting) ---")

        # Ortak örnek indekslerini belirle (kesinlik için)
        common_indices = set(predictions[available_models[0]]['indices'])
        for name in available_models[1:]:
            common_indices &= set(predictions[name]['indices'])

        if common_indices:
            common_indices = sorted(list(common_indices))
            ensemble_preds = []
            ensemble_true = []

            # Ağırlıklar manuel olarak belirlenmiştir (toplamı 1 olmalı)
            model_weights = {
                'XGBoost': 0.2,
                'RandomForest': 0.3,
                'LSTM': 0.5
            }

            # Her modelin index -> prediction haritası
            model_preds_by_index = {
                name: {idx: pred for idx, pred in zip(data['indices'], data['pred'])}
                for name, data in predictions.items()
            }

            for idx in common_indices:
                weighted_votes = defaultdict(float)
                for model_name in available_models:
                    pred = model_preds_by_index[model_name][idx]
                    weighted_votes[pred] += model_weights.get(model_name, 1.0)
                final_pred = max(weighted_votes, key=weighted_votes.get)
                ensemble_preds.append(final_pred)
                ensemble_true.append(model_preds_by_index[available_models[0]][idx])  # hepsi aynı

            ensemble_preds = np.array(ensemble_preds)
            ensemble_true = np.array(ensemble_true)

            pred_time = sum([predictions[name]['time'] for name in available_models])

            predictions['Ensemble'] = {
                'pred': ensemble_preds,
                'true': ensemble_true,
                'indices': common_indices,
                'time': pred_time
            }

            print(f"Prediction time (summed): {pred_time:.3f} seconds")
        else:
            print("❌ No overlapping indices among base models. Skipping Ensemble.")

    # Generate detailed reports and visualizations
    results_data = []
    label_to_cell_id = {label: cell_id for label, cell_id in enumerate(label_encoder.classes_)}
    
    # Create subplots for error distribution
    fig, axes = plt.subplots(1, len(predictions), figsize=(6*len(predictions), 5))
    if len(predictions) == 1:
        axes = [axes]
    
    for idx, (name, data) in enumerate(predictions.items()):
        y_pred = data['pred']
        y_true = data['true']
        eval_indices = data['indices']
        pred_time = data['time']
        
        print(f"\n{'='*50}")
        print(f"--- Detailed Evaluation Report for {name} ---")
        print(f"{'='*50}")
        
        # Classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Get unique labels for the report
        unique_labels = np.unique(np.concatenate((y_true, y_pred)))
        
        # Create classification report with limited classes for readability
        if len(unique_labels) > 10:
            print(f"\nShowing metrics for top 10 classes (out of {len(unique_labels)} total):")
            top_classes = pd.Series(y_true).value_counts().head(10).index
            mask = np.isin(y_true, top_classes) & np.isin(y_pred, top_classes)
            report = classification_report(y_true[mask], y_pred[mask], zero_division=0, digits=4)
        else:
            report = classification_report(y_true, y_pred, zero_division=0, digits=4)
        
        print(report)
        
        # Calculate location errors
        true_coords = df_filtered.loc[eval_indices][['Latitude', 'Longitude']].values
        pred_cell_ids = [label_to_cell_id.get(label) for label in y_pred]
        
        valid_indices = [i for i, cid in enumerate(pred_cell_ids) if cid is not None and cid in cell_centers]
        if len(valid_indices) > 0:
            pred_coords = np.array([cell_centers[pred_cell_ids[i]] for i in valid_indices])
            
            distances = haversine_distance(
                true_coords[valid_indices, 0], true_coords[valid_indices, 1], 
                pred_coords[:, 0], pred_coords[:, 1]
            )
            
            mean_error = np.mean(distances)
            median_error = np.median(distances)
            std_error = np.std(distances)
            percentile_90 = np.percentile(distances, 90)
            
            print(f"\nLocation Error Statistics:")
            print(f"  Mean Error: {mean_error:.2f} meters")
            print(f"  Median Error: {median_error:.2f} meters")
            print(f"  Std Dev: {std_error:.2f} meters")
            print(f"  90th Percentile: {percentile_90:.2f} meters")
            print(f"  Min Error: {np.min(distances):.2f} meters")
            print(f"  Max Error: {np.max(distances):.2f} meters")
            
            # Plot error distribution
            axes[idx].hist(distances, bins=50, alpha=0.7, color=plt.cm.tab10(idx))
            axes[idx].axvline(mean_error, color='red', linestyle='--', label=f'Mean: {mean_error:.1f}m')
            axes[idx].axvline(median_error, color='green', linestyle='--', label=f'Median: {median_error:.1f}m')
            axes[idx].set_xlabel('Error Distance (meters)')
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(f'{name} Error Distribution')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
            
            # Save confusion matrix
            cm_path = os.path.join(visuals_dir, f'{name}_confusion_matrix.png')
            plot_confusion_matrix(y_true, y_pred, name, cm_path)
            
        else:
            mean_error = np.inf
            median_error = np.inf
            percentile_90 = np.inf
            print("\nWarning: No valid predictions for error calculation")
        
        results_data.append({
            'Model': name, 
            'Accuracy (%)': accuracy * 100, 
            'Mean Error (m)': mean_error,
            'Median Error (m)': median_error,
            '90th Percentile (m)': percentile_90,
            'Inference Time (s)': pred_time
        })

    plt.tight_layout()
    plt.savefig(os.path.join(visuals_dir, 'error_distributions.png'), dpi=300)
    plt.close()

    # Create comparison visualizations
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Prediction paths comparison
    plt.subplot(2, 2, 1)
    colors = {'XGBoost': 'red', 'RandomForest': 'green', 'LSTM': 'purple'}
    
    # Plot a subset of predictions for clarity
    n_points = min(500, len(test_indices_tree))
    for name, data in predictions.items():
        eval_indices = data['indices']
        y_pred = data['pred']
        pred_cell_ids = [label_to_cell_id.get(label) for label in y_pred[:n_points]]
        valid_indices = [i for i, cid in enumerate(pred_cell_ids) if cid is not None and cid in cell_centers]
        if valid_indices:
            pred_coords = np.array([cell_centers[pred_cell_ids[i]] for i in valid_indices])
            plt.scatter(pred_coords[:, 1], pred_coords[:, 0], 
                       color=colors.get(name, 'blue'), 
                       label=f'{name} Predictions', 
                       s=20, alpha=0.6)
    
    # Plot actual path
    if 'XGBoost' in predictions:
        true_coords = df_filtered.loc[predictions['XGBoost']['indices'][:n_points]][['Latitude', 'Longitude']].values
        plt.scatter(true_coords[:, 1], true_coords[:, 0], 
                   color='black', label='Actual Path', s=10, alpha=0.9, zorder=5)
    
    plt.title('Model Predictions vs Actual Path')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Performance metrics comparison
    results_df = pd.DataFrame(results_data).round(2)
    
    plt.subplot(2, 2, 2)
    metrics = ['Accuracy (%)', 'Mean Error (m)', 'Median Error (m)']
    x = np.arange(len(results_df))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        plt.bar(x + i*width, results_df[metric], width, label=metric)
    
    plt.xlabel('Model')
    plt.ylabel('Value')
    plt.title('Performance Metrics Comparison')
    plt.xticks(x + width, results_df['Model'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Error percentiles
    plt.subplot(2, 2, 3)
    models = results_df['Model'].values
    mean_errors = results_df['Mean Error (m)'].values
    median_errors = results_df['Median Error (m)'].values
    p90_errors = results_df['90th Percentile (m)'].values
    
    x = np.arange(len(models))
    plt.plot(x, mean_errors, 'o-', label='Mean', markersize=10)
    plt.plot(x, median_errors, 's-', label='Median', markersize=10)
    plt.plot(x, p90_errors, '^-', label='90th Percentile', markersize=10)
    
    plt.xlabel('Model')
    plt.ylabel('Error (meters)')
    plt.title('Error Statistics by Model')
    plt.xticks(x, models)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Inference time
    plt.subplot(2, 2, 4)
    plt.bar(results_df['Model'], results_df['Inference Time (s)'])
    plt.xlabel('Model')
    plt.ylabel('Time (seconds)')
    plt.title('Inference Time Comparison')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(visuals_dir, 'model_comparison_detailed.png'), dpi=300)
    plt.show()

    # Print final summary
    print("\n" + "="*70)
    print("FINAL PERFORMANCE SUMMARY")
    print("="*70)
    print(results_df.to_string(index=False))
    
    # Save results to CSV
    results_df.to_csv(os.path.join(visuals_dir, 'model_performance_results.csv'), index=False)
    print(f"\n✅ Results saved to {os.path.join(visuals_dir, 'model_performance_results.csv')}")
    
    # Identify best model
    best_accuracy_model = results_df.loc[results_df['Accuracy (%)'].idxmax(), 'Model']
    best_error_model = results_df.loc[results_df['Mean Error (m)'].idxmin(), 'Model']
    
    print(f"\n🏆 Best Accuracy: {best_accuracy_model}")
    print(f"🎯 Lowest Mean Error: {best_error_model}")
    
    # Competition readiness assessment
    print("\n" + "="*70)
    print("COMPETITION READINESS ASSESSMENT")
    print("="*70)
    
    for _, row in results_df.iterrows():
        model = row['Model']
        mean_error = row['Mean Error (m)']
        inference_time = row['Inference Time (s)']
        
        print(f"\n{model}:")
        if mean_error < 15:
            print("  ✅ Excellent accuracy for competition (< 15m)")
        elif mean_error < 30:
            print("  ⚠️  Good accuracy, consider improvements (15-30m)")
        else:
            print("  ❌ Needs improvement for competition (> 30m)")
            
        if inference_time < 1:
            print("  ✅ Fast inference suitable for real-time")
        else:
            print("  ⚠️  Consider optimization for faster inference")


# Execution
if __name__ == '__main__':
    import time
    DATA_PATH = os.path.join("Data", "Merged_encoded_filled_filtered_pciCleaned_featureEngineered.xlsx")
    evaluate_models(DATA_PATH)