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
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
    matthews_corrcoef, cohen_kappa_score, 
    balanced_accuracy_score, top_k_accuracy_score
)
from tensorflow.keras.models import load_model
import tensorflow as tf
import geopandas as gpd
from scipy import stats
import json

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

def calculate_comprehensive_location_metrics(true_coords, pred_coords, timestamps=None):
    """
    Calculate comprehensive location-specific metrics for positioning accuracy
    """
    distances = haversine_distance(
        true_coords[:, 0], true_coords[:, 1],
        pred_coords[:, 0], pred_coords[:, 1]
    )
    
    metrics = {
        # Basic Statistics
        'mean_error': np.mean(distances),
        'median_error': np.median(distances),
        'std_error': np.std(distances),
        'rmse': np.sqrt(np.mean(distances**2)),
        'mae': np.mean(np.abs(distances)),
        
        # Percentile Analysis
        'percentile_50': np.percentile(distances, 50),
        'percentile_67': np.percentile(distances, 67),  # 2D-CEP
        'percentile_75': np.percentile(distances, 75),
        'percentile_90': np.percentile(distances, 90),
        'percentile_95': np.percentile(distances, 95),
        'percentile_99': np.percentile(distances, 99),
        
        # CEP (Circular Error Probable) - Industry Standard
        'cep50': np.percentile(distances, 50),  # 50% of points within this radius
        'cep67': np.percentile(distances, 67),  # ~1 std dev for 2D normal
        'cep95': np.percentile(distances, 95),  # ~2 std dev for 2D normal
        'r95': np.percentile(distances, 95),    # 95% radius
        
        # Outlier Analysis
        'outlier_ratio_15m': np.sum(distances > 15) / len(distances),
        'outlier_ratio_30m': np.sum(distances > 30) / len(distances),
        'outlier_ratio_50m': np.sum(distances > 50) / len(distances),
        
        # Success Rate at Different Thresholds
        'success_rate_5m': np.sum(distances <= 5) / len(distances),
        'success_rate_10m': np.sum(distances <= 10) / len(distances),
        'success_rate_15m': np.sum(distances <= 15) / len(distances),
        'success_rate_20m': np.sum(distances <= 20) / len(distances),
        
        # Distribution Metrics
        'skewness': stats.skew(distances),
        'kurtosis': stats.kurtosis(distances),
        'iqr': np.percentile(distances, 75) - np.percentile(distances, 25),
    }
    
    return metrics, distances

def calculate_advanced_classification_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calculate advanced classification metrics beyond basic accuracy
    """
    metrics = {
        # Balanced metrics for imbalanced classes
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        
        # Agreement metrics
        'matthews_correlation': matthews_corrcoef(y_true, y_pred),
        'cohen_kappa': cohen_kappa_score(y_true, y_pred),
        
        # Per-class metrics
        'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    # Top-K accuracy if probabilities are available
    if y_pred_proba is not None:
        try:
            metrics['top_3_accuracy'] = top_k_accuracy_score(y_true, y_pred_proba, k=3)
            metrics['top_5_accuracy'] = top_k_accuracy_score(y_true, y_pred_proba, k=5)
        except:
            pass
    
    return metrics

def analyze_error_distribution(distances):
    """
    Detailed analysis of error distribution characteristics
    """
    # Fit different distributions
    distributions = {
        'normal': stats.norm,
        'lognormal': stats.lognorm,
        'gamma': stats.gamma,
        'weibull': stats.weibull_min,
        'rayleigh': stats.rayleigh
    }
    
    best_fit = None
    best_ks_stat = np.inf
    
    for name, dist in distributions.items():
        try:
            if name == 'lognormal':
                shape, loc, scale = dist.fit(distances, floc=0)
                params = (shape, loc, scale)
            else:
                params = dist.fit(distances)
            
            ks_stat, p_value = stats.kstest(distances, lambda x: dist.cdf(x, *params))
            
            if ks_stat < best_ks_stat:
                best_ks_stat = ks_stat
                best_fit = (name, params, p_value)
        except:
            continue
    
    # Calculate confidence intervals
    ci_95 = np.percentile(distances, [2.5, 97.5])
    ci_99 = np.percentile(distances, [0.5, 99.5])
    
    # Calculate entropy
    hist, _ = np.histogram(distances, bins=50)
    hist_normalized = hist / np.sum(hist)
    hist_normalized = hist_normalized[hist_normalized > 0]
    entropy = -np.sum(hist_normalized * np.log(hist_normalized))
    
    return {
        'best_fit_distribution': best_fit[0] if best_fit else None,
        'ks_test_pvalue': best_fit[2] if best_fit else None,
        'ci_95_lower': ci_95[0],
        'ci_95_upper': ci_95[1],
        'ci_99_lower': ci_99[0],
        'ci_99_upper': ci_99[1],
        'error_entropy': entropy
    }

def create_comprehensive_visualizations(model_name, distances, all_metrics, visuals_dir):
    """
    Create comprehensive visualization plots for detailed analysis
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. CDF Plot
    ax = axes[0, 0]
    sorted_distances = np.sort(distances)
    cdf = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances)
    ax.plot(sorted_distances, cdf, 'b-', linewidth=2)
    ax.axvline(all_metrics['cep50'], color='r', linestyle='--', label=f'CEP50: {all_metrics["cep50"]:.1f}m')
    ax.axvline(all_metrics['cep95'], color='g', linestyle='--', label=f'CEP95: {all_metrics["cep95"]:.1f}m')
    ax.set_xlabel('Error Distance (m)')
    ax.set_ylabel('CDF')
    ax.set_title('Cumulative Distribution Function')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Success Rate by Threshold
    ax = axes[0, 1]
    thresholds = [5, 10, 15, 20, 30, 50]
    success_rates = [all_metrics[f'success_rate_{t}m'] * 100 for t in [5, 10, 15, 20]]
    success_rates.extend([
        (1 - all_metrics['outlier_ratio_30m']) * 100,
        (1 - all_metrics['outlier_ratio_50m']) * 100
    ])
    ax.bar(range(len(thresholds)), success_rates, color='skyblue', edgecolor='navy')
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([f'{t}m' for t in thresholds])
    ax.set_xlabel('Distance Threshold')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate at Different Thresholds')
    ax.grid(True, axis='y', alpha=0.3)
    
    # 3. Q-Q Plot for distribution analysis
    ax = axes[0, 2]
    stats.probplot(distances, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (Normal Distribution)')
    
    # 4. Error Percentiles
    ax = axes[1, 0]
    percentiles = [50, 67, 75, 90, 95, 99]
    percentile_values = [all_metrics[f'percentile_{p}'] for p in percentiles]
    ax.plot(percentiles, percentile_values, 'o-', markersize=8, linewidth=2)
    ax.set_xlabel('Percentile')
    ax.set_ylabel('Error Distance (m)')
    ax.set_title('Error Percentiles')
    ax.grid(True, alpha=0.3)
    
    # 5. Distribution Histogram with fitted curve
    ax = axes[1, 1]
    n, bins, _ = ax.hist(distances, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
    
    # Add fitted distribution if available
    if all_metrics.get('best_fit_distribution'):
        dist_name = all_metrics['best_fit_distribution']
        x = np.linspace(0, max(distances), 100)
        if dist_name == 'normal':
            mu, sigma = np.mean(distances), np.std(distances)
            ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
                   label=f'Fitted {dist_name}')
        elif dist_name == 'rayleigh':
            param = stats.rayleigh.fit(distances)
            ax.plot(x, stats.rayleigh.pdf(x, *param), 'r-', linewidth=2, 
                   label=f'Fitted {dist_name}')
    
    ax.set_xlabel('Error Distance (m)')
    ax.set_ylabel('Density')
    ax.set_title('Error Distribution')
    ax.legend()
    
    # 6. Box Plot for error distribution
    ax = axes[1, 2]
    box_data = [distances[distances <= 20], distances[(distances > 20) & (distances <= 50)], 
                distances[distances > 50]] if any(distances > 50) else [distances]
    labels = ['≤20m', '20-50m', '>50m'] if any(distances > 50) else ['All']
    ax.boxplot(box_data, labels=labels)
    ax.set_ylabel('Error Distance (m)')
    ax.set_title('Error Distribution by Range')
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.suptitle(f'Comprehensive Error Analysis - {model_name} (Weighted)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(visuals_dir, f'{model_name}_comprehensive_analysis_weighted.png'), dpi=300)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    """
    Create and save a confusion matrix plot.
    """
    unique_classes = np.unique(np.concatenate((y_true, y_pred)))
    if len(unique_classes) > 20:
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
    Loads trained models and evaluates their performance using probability-weighted
    coordinate calculation with comprehensive analysis and visualizations.
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
            start_time = time.time()
            y_pred_proba = model.predict_proba(X_test_tree)
            pred_time = time.time() - start_time
            predictions[model_name] = {
                'pred_proba': y_pred_proba, 
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
        pred_time = time.time() - start_time
        predictions['LSTM'] = {
            'pred_proba': y_pred_proba, 
            'true': y_test_lstm, 
            'indices': test_indices_lstm,
            'time': pred_time
        }
        print(f"Prediction time: {pred_time:.3f} seconds")

    # --- 5. Ensemble Model ---
    model_weights = {'XGBoost': 0.2, 'RandomForest': 0.3, 'LSTM': 0.5}
    available_models = [name for name in model_weights if name in predictions]

    if len(available_models) >= 2:
        print("\n--- Creating Ensemble Model ---")
        common_indices = set(predictions[available_models[0]]['indices'])
        for name in available_models[1:]:
            common_indices &= set(predictions[name]['indices'])

        common_indices = sorted(list(common_indices))
        if common_indices:
            ensemble_proba = []
            ensemble_true = []
            
            # Create index mapping for each model
            model_index_maps = {}
            for model_name in available_models:
                indices = predictions[model_name]['indices']
                if hasattr(indices, 'get_loc'):  # pandas Index
                    model_index_maps[model_name] = {idx: indices.get_loc(idx) for idx in common_indices}
                else:  # numpy array
                    model_index_maps[model_name] = {idx: np.where(indices == idx)[0][0] for idx in common_indices}
            
            for idx in common_indices:
                combined = np.zeros_like(predictions[available_models[0]]['pred_proba'][0])
                for model_name in available_models:
                    model_idx = model_index_maps[model_name][idx]
                    combined += model_weights[model_name] * predictions[model_name]['pred_proba'][model_idx]
                ensemble_proba.append(combined)
            
            # Get true labels for common indices
            true_indices = predictions[available_models[0]]['indices']
            if hasattr(true_indices, 'get_loc'):
                ensemble_true = [predictions[available_models[0]]['true'][true_indices.get_loc(idx)] for idx in common_indices]
            else:
                ensemble_true = [predictions[available_models[0]]['true'][np.where(true_indices == idx)[0][0]] for idx in common_indices]
            
            ensemble_proba = np.array(ensemble_proba)
            ensemble_true = np.array(ensemble_true)
            
            pred_time = sum([predictions[name]['time'] for name in available_models])
            
            predictions['Ensemble'] = {
                'pred_proba': ensemble_proba,
                'true': ensemble_true,
                'indices': common_indices,
                'time': pred_time
            }
            print(f"Ensemble created with {len(common_indices)} common samples")

    # --- 6. Comprehensive evaluation for each model ---
    results_data = []
    comprehensive_results = {}
    
    # Create subplots for error distributions
    fig, axes = plt.subplots(1, len(predictions), figsize=(6*len(predictions), 5), squeeze=False)
    
    for idx, (name, data) in enumerate(predictions.items()):
        ax = axes[0, idx]
        print(f"\n{'='*70}")
        print(f"COMPREHENSIVE EVALUATION FOR {name}")
        print(f"{'='*70}")
        
        y_pred_proba = data['pred_proba']
        y_true = data['true']
        eval_indices = data['indices']
        pred_time = data['time']
        
        # Classification metrics (based on top-1 prediction)
        y_pred_top1 = np.argmax(y_pred_proba, axis=1)
        
        # Calculate advanced classification metrics
        class_metrics = calculate_advanced_classification_metrics(y_true, y_pred_top1, y_pred_proba)
        accuracy = accuracy_score(y_true, y_pred_top1)
        
        # Location Error Calculation (using weighted method)
        true_coords = df_filtered.loc[eval_indices][['Latitude', 'Longitude']].values
        
        # Use different top_n values for analysis
        top_n_values = [1, 3, 5, 10]
        weighted_results = {}
        
        for top_n in top_n_values:
            pred_coords_weighted = calculate_weighted_coordinates(y_pred_proba, label_encoder, cell_centers, top_n=top_n)
            valid_indices = ~np.isnan(pred_coords_weighted).any(axis=1)
            
            if np.any(valid_indices):
                distances = haversine_distance(
                    true_coords[valid_indices, 0], true_coords[valid_indices, 1], 
                    pred_coords_weighted[valid_indices, 0], pred_coords_weighted[valid_indices, 1]
                )
                weighted_results[f'top_{top_n}'] = {
                    'mean_error': np.mean(distances),
                    'median_error': np.median(distances),
                    'percentile_90': np.percentile(distances, 90)
                }
        
        # Use top-5 for main analysis
        pred_coords_weighted = calculate_weighted_coordinates(y_pred_proba, label_encoder, cell_centers, top_n=5)
        valid_indices = ~np.isnan(pred_coords_weighted).any(axis=1)
        
        if np.any(valid_indices):
            # Calculate comprehensive location metrics
            location_metrics, distances = calculate_comprehensive_location_metrics(
                true_coords[valid_indices], pred_coords_weighted[valid_indices]
            )
            
            # Error distribution analysis
            dist_metrics = analyze_error_distribution(distances)
            
            # Combine all metrics
            comprehensive_results[name] = {
                **location_metrics,
                **class_metrics,
                **dist_metrics,
                'weighted_results': weighted_results,
                'accuracy': accuracy
            }
            
            # Print results
            print(f"\n📊 CLASSIFICATION METRICS:")
            print(f"  Top-1 Accuracy: {accuracy*100:.2f}%")
            print(f"  Top-3 Accuracy: {class_metrics.get('top_3_accuracy', 0)*100:.2f}%")
            print(f"  Top-5 Accuracy: {class_metrics.get('top_5_accuracy', 0)*100:.2f}%")
            print(f"  Balanced Accuracy: {class_metrics['balanced_accuracy']*100:.2f}%")
            print(f"  Matthews Correlation: {class_metrics['matthews_correlation']:.3f}")
            
            print(f"\n📍 LOCATION ACCURACY (Weighted Top-5):")
            print(f"  Mean Error: {location_metrics['mean_error']:.2f} meters")
            print(f"  Median Error: {location_metrics['median_error']:.2f} meters")
            print(f"  RMSE: {location_metrics['rmse']:.2f} meters")
            print(f"  CEP50: {location_metrics['cep50']:.2f} meters")
            print(f"  CEP95: {location_metrics['cep95']:.2f} meters")
            
            print(f"\n🎯 SUCCESS RATES:")
            print(f"  Within 5m: {location_metrics['success_rate_5m']*100:.1f}%")
            print(f"  Within 10m: {location_metrics['success_rate_10m']*100:.1f}%")
            print(f"  Within 15m: {location_metrics['success_rate_15m']*100:.1f}%")
            
            print(f"\n🔍 WEIGHTED ANALYSIS (Different Top-N):")
            for top_n, results in weighted_results.items():
                print(f"  {top_n}: Mean={results['mean_error']:.2f}m, Median={results['median_error']:.2f}m")
            
            # Plot error distribution
            ax.hist(distances, bins=50, alpha=0.7, color=plt.cm.tab10(idx))
            ax.axvline(location_metrics['mean_error'], color='red', linestyle='--', 
                      label=f'Mean: {location_metrics["mean_error"]:.1f}m')
            ax.axvline(location_metrics['median_error'], color='green', linestyle='--', 
                      label=f'Median: {location_metrics["median_error"]:.1f}m')
            ax.set_xlabel('Error Distance (meters)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{name} Error Distribution (Weighted)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Create comprehensive visualizations
            create_comprehensive_visualizations(name, distances, comprehensive_results[name], visuals_dir)
            
            # Save confusion matrix
            cm_path = os.path.join(visuals_dir, f'{name}_confusion_matrix_weighted.png')
            plot_confusion_matrix(y_true, y_pred_top1, name, cm_path)
            
            # Add to results summary
            results_data.append({
                'Model': name,
                'Top-1 Acc (%)': accuracy * 100,
                'Top-3 Acc (%)': class_metrics.get('top_3_accuracy', 0) * 100,
                'Balanced Acc (%)': class_metrics['balanced_accuracy'] * 100,
                'Mean Error (m)': location_metrics['mean_error'],
                'Median Error (m)': location_metrics['median_error'],
                'RMSE (m)': location_metrics['rmse'],
                'CEP50 (m)': location_metrics['cep50'],
                'CEP95 (m)': location_metrics['cep95'],
                'Success <10m (%)': location_metrics['success_rate_10m'] * 100,
                'Success <15m (%)': location_metrics['success_rate_15m'] * 100,
                'Inference Time (s)': pred_time
            })
            
        else:
            print("\nWarning: No valid predictions for error calculation")

    plt.tight_layout()
    plt.savefig(os.path.join(visuals_dir, 'error_distributions_weighted.png'), dpi=300)
    plt.close()

    # --- 7. Create comprehensive comparison visualizations ---
    if results_data:
        results_df = pd.DataFrame(results_data).round(2)
        
        # Create detailed comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Accuracy Metrics Comparison
        ax = axes[0, 0]
        acc_metrics = ['Top-1 Acc (%)', 'Top-3 Acc (%)', 'Balanced Acc (%)', 'Success <10m (%)', 'Success <15m (%)']
        results_melted = results_df.melt(id_vars='Model', value_vars=acc_metrics)
        sns.barplot(data=results_melted, x='variable', y='value', hue='Model', ax=ax)
        ax.set_title('Accuracy Metrics Comparison (Weighted)')
        ax.set_xlabel('')
        ax.set_ylabel('Percentage (%)')
        ax.tick_params(axis='x', rotation=15)
        
        # Plot 2: Error Metrics Comparison
        ax = axes[0, 1]
        error_metrics = ['Mean Error (m)', 'Median Error (m)', 'RMSE (m)']
        results_melted = results_df.melt(id_vars='Model', value_vars=error_metrics)
        sns.barplot(data=results_melted, x='variable', y='value', hue='Model', ax=ax)
        ax.set_title('Error Metrics Comparison (Weighted)')
        ax.set_xlabel('')
        ax.set_ylabel('Distance (meters)')
        ax.tick_params(axis='x', rotation=15)
        
        # Plot 3: CEP Comparison
        ax = axes[1, 0]
        models = results_df['Model'].values
        cep50 = results_df['CEP50 (m)'].values
        cep95 = results_df['CEP95 (m)'].values
        
        x = np.arange(len(models))
        width = 0.35
        ax.bar(x - width/2, cep50, width, label='CEP50')
        ax.bar(x + width/2, cep95, width, label='CEP95')
        ax.set_xlabel('Model')
        ax.set_ylabel('Distance (meters)')
        ax.set_title('Circular Error Probable Comparison (Weighted)')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        # Plot 4: Model Performance Radar Chart
        ax = axes[1, 1]
        categories = ['Accuracy', 'Speed', 'Precision', 'Robustness']
        
        for i, row in results_df.iterrows():
            model = row['Model']
            accuracy = row['Top-1 Acc (%)'] / 100
            speed = 1 - (row['Inference Time (s)'] / results_df['Inference Time (s)'].max())
            precision = 1 - (row['Mean Error (m)'] / results_df['Mean Error (m)'].max())
            robustness = row['Success <15m (%)'] / 100
            
            values = [accuracy, speed, precision, robustness]
            values += values[:1]  # Complete the circle
            
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
            angles = np.concatenate([angles, [angles[0]]])
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model)
            ax.fill(angles, values, alpha=0.15)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Overview (Weighted)')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(visuals_dir, 'comprehensive_comparison_weighted.png'), dpi=300)
        plt.show()

        # Print final summary
        print("\n" + "="*100)
        print("FINAL PERFORMANCE SUMMARY (PROBABILITY WEIGHTED)")
        print("="*100)
        print(results_df.to_string(index=False))
        
        # Save results
        results_df.to_csv(os.path.join(visuals_dir, 'comprehensive_performance_weighted.csv'), index=False)
        
        # Save detailed metrics to JSON
        with open(os.path.join(visuals_dir, 'detailed_metrics_weighted.json'), 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        
        print(f"\n✅ All results saved to {visuals_dir}")
        
        # Identify best models
        best_accuracy_model = results_df.loc[results_df['Top-1 Acc (%)'].idxmax(), 'Model']
        best_error_model = results_df.loc[results_df['Mean Error (m)'].idxmin(), 'Model']
        best_cep50_model = results_df.loc[results_df['CEP50 (m)'].idxmin(), 'Model']
        
        print(f"\n🏆 BEST PERFORMERS (Weighted Approach):")
        print(f"  Highest Accuracy: {best_accuracy_model}")
        print(f"  Lowest Mean Error: {best_error_model}")
        print(f"  Best CEP50: {best_cep50_model}")
        
        # Create Top-N comparison plot
        plt.figure(figsize=(12, 8))
        top_n_comparison = []
        
        for model_name, metrics in comprehensive_results.items():
            if 'weighted_results' in metrics:
                for top_n, results in metrics['weighted_results'].items():
                    top_n_comparison.append({
                        'Model': model_name,
                        'Top-N': top_n,
                        'Mean Error': results['mean_error']
                    })
        
        if top_n_comparison:
            top_n_df = pd.DataFrame(top_n_comparison)
            sns.lineplot(data=top_n_df, x='Top-N', y='Mean Error', hue='Model', marker='o', markersize=8)
            plt.title('Mean Error vs Top-N Weighted Cells')
            plt.xlabel('Number of Top Cells Used')
            plt.ylabel('Mean Error (meters)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(visuals_dir, 'top_n_comparison.png'), dpi=300)
            plt.show()
        plot_scatter_predictions(df_filtered, predictions, label_encoder, cell_centers, visuals_dir)

def plot_scatter_predictions(df_filtered, predictions, label_encoder, cell_centers, save_dir):
    """
    Draw scatter plots comparing predicted vs. true coordinates (Weighted Prediction).
    """
    label_to_cell_id = {label: cell_id for label, cell_id in enumerate(label_encoder.classes_)}

    for model_name, data in predictions.items():
        y_pred_proba = data['pred_proba']
        indices = data['indices']
        true_coords = df_filtered.loc[indices][['Latitude', 'Longitude']].values
        
        pred_coords_weighted = calculate_weighted_coordinates(y_pred_proba, label_encoder, cell_centers, top_n=5)
        valid_indices = ~np.isnan(pred_coords_weighted).any(axis=1)

        if not np.any(valid_indices):
            continue

        true_valid = true_coords[valid_indices]
        pred_valid = pred_coords_weighted[valid_indices]

        plt.figure(figsize=(8, 8))
        plt.scatter(true_valid[:, 1], true_valid[:, 0], c='green', s=20, label='True Location', alpha=0.6)
        plt.scatter(pred_valid[:, 1], pred_valid[:, 0], c='red', s=20, label='Predicted Location', alpha=0.6)
        plt.title(f"Prediction vs True Location - {model_name} (Weighted)")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{model_name}_scatter_true_vs_pred_weighted.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

if __name__ == '__main__':
    DATA_PATH = os.path.join("Data", "Merged_encoded_filled_filtered_pciCleaned_featureEngineered.xlsx")
    evaluate_models(DATA_PATH)