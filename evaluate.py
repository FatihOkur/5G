import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
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
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import geopandas as gpd
from scipy import stats

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

def calculate_5g_specific_metrics(df_test, y_pred, y_true, label_to_cell_id, cell_centers):
    """
    Calculate metrics specific to 5G signal conditions
    """
    metrics = {}
    
    # Calculate distances for this subset
    true_coords = df_test[['Latitude', 'Longitude']].values
    pred_cell_ids = [label_to_cell_id.get(label) for label in y_pred]
    valid_indices = [i for i, cid in enumerate(pred_cell_ids) if cid is not None and cid in cell_centers]
    
    if len(valid_indices) == 0:
        return metrics
    
    pred_coords = np.array([cell_centers[pred_cell_ids[i]] for i in valid_indices])
    distances = haversine_distance(
        true_coords[valid_indices, 0], true_coords[valid_indices, 1],
        pred_coords[:, 0], pred_coords[:, 1]
    )
    
    # Performance by RSRP levels
    if 'RSRP' in df_test.columns:
        rsrp_bins = [-120, -100, -90, -80, -70, -50]
        df_test['rsrp_category'] = pd.cut(df_test['RSRP'], bins=rsrp_bins, 
                                          labels=['Very Poor', 'Poor', 'Fair', 'Good', 'Excellent'])
        
        for category in df_test['rsrp_category'].dropna().unique():
            mask = df_test['rsrp_category'] == category
            valid_mask = np.array([i for i, m in enumerate(mask) if m and i in valid_indices])
            if len(valid_mask) > 0:
                cat_distances = distances[np.isin(valid_indices, valid_mask)]
                metrics[f'mean_error_rsrp_{category}'] = np.mean(cat_distances)
                metrics[f'accuracy_rsrp_{category}'] = accuracy_score(y_true[mask], y_pred[mask])
    
    # Performance by SINR levels
    if 'SINR' in df_test.columns:
        sinr_bins = [-10, 0, 10, 20, 30, 40]
        df_test['sinr_category'] = pd.cut(df_test['SINR'], bins=sinr_bins,
                                          labels=['Very Poor', 'Poor', 'Fair', 'Good', 'Excellent'])
        
        for category in df_test['sinr_category'].dropna().unique():
            mask = df_test['sinr_category'] == category
            valid_mask = np.array([i for i, m in enumerate(mask) if m and i in valid_indices])
            if len(valid_mask) > 0:
                cat_distances = distances[np.isin(valid_indices, valid_mask)]
                metrics[f'mean_error_sinr_{category}'] = np.mean(cat_distances)
                metrics[f'accuracy_sinr_{category}'] = accuracy_score(y_true[mask], y_pred[mask])
    
    return metrics

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

def calculate_spatial_coherence_metrics(true_coords, pred_coords, timestamps=None):
    """
    Evaluate how spatially coherent predictions are (important for trajectory)
    """
    if timestamps is None or len(timestamps) < 2:
        return {}
    
    # Sort by timestamp
    sorted_idx = np.argsort(timestamps)
    true_sorted = true_coords[sorted_idx]
    pred_sorted = pred_coords[sorted_idx]
    
    # Calculate trajectory distances
    true_distances = np.sqrt(np.sum(np.diff(true_sorted, axis=0)**2, axis=1))
    pred_distances = np.sqrt(np.sum(np.diff(pred_sorted, axis=0)**2, axis=1))
    
    # Avoid division by zero
    if np.std(true_distances) == 0:
        trajectory_smoothness = np.inf if np.std(pred_distances) > 0 else 1.0
    else:
        trajectory_smoothness = np.std(pred_distances) / np.std(true_distances)
    
    # Direction changes
    true_directions = np.arctan2(np.diff(true_sorted[:, 1]), np.diff(true_sorted[:, 0]))
    pred_directions = np.arctan2(np.diff(pred_sorted[:, 1]), np.diff(pred_sorted[:, 0]))
    
    direction_errors = np.abs(true_directions - pred_directions)
    direction_errors = np.minimum(direction_errors, 2*np.pi - direction_errors)
    
    metrics = {
        'trajectory_smoothness': trajectory_smoothness,
        'mean_direction_error': np.mean(direction_errors) * 180 / np.pi,  # degrees
        'trajectory_length_ratio': np.sum(pred_distances) / np.sum(true_distances) if np.sum(true_distances) > 0 else np.inf,
        'max_jump': np.max(pred_distances),
    }
    
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
                # lognorm needs special handling
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
    hist_normalized = hist_normalized[hist_normalized > 0]  # Remove zeros
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
    
    plt.suptitle(f'Comprehensive Error Analysis - {model_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(visuals_dir, f'{model_name}_comprehensive_analysis.png'), dpi=300)
    plt.close()

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

def label_land_type_by_buffer(df, lat_col='Latitude', lon_col='Longitude', buffer_radius=5):
    """
    Her noktanın çevresindeki buffer alanı içinde hangi coğrafi yapı en yoğunsa onu 'land_type' olarak etiketler.
    """
    terrain_dir = "ITUMapData"
    metric_crs = "EPSG:3857"
    geo_crs = "EPSG:4326"

    # Geo veri setlerini yükle
    building = gpd.read_file(f"{terrain_dir}/ITU_3DBINA_EPSG4326.shp").to_crs(metric_crs)
    vegetation = gpd.read_file(f"{terrain_dir}/ITU_3DVEGETATION_EPSG4326.shp").to_crs(metric_crs)
    road = gpd.read_file(f"{terrain_dir}/ITU_ULASIMAGI_EPSG4326.shp").to_crs(metric_crs)
    water = gpd.read_file(f"{terrain_dir}/ITU_SUKUTLESI_EPSG4326.shp").to_crs(metric_crs)
    wall = gpd.read_file(f"{terrain_dir}/ITU_SINIRDUVAR_EPSG4326.shp").to_crs(metric_crs)

    # Noktaları GeoDataFrame'e çevir
    gdf = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs=geo_crs)
    gdf = gdf.to_crs(metric_crs)

    # 5 metrelik buffer alan oluştur
    gdf['buffer'] = gdf.geometry.buffer(buffer_radius)

    # Coğrafi yapıların yoğunluklarını hesapla
    sources = {
        'building': building,
        'vegetation': vegetation,
        'road': road,
        'water': water,
        'wall': wall
    }

    def get_dominant_type(buffer_geom):
        densities = {}
        for lt, src in sources.items():
            clipped = src.clip(buffer_geom)
            if clipped.empty:
                densities[lt] = 0
            else:
                # Yol ve duvar çizgi olduğu için uzunluk, diğerleri alan
                if clipped.geometry.iloc[0].geom_type in ['LineString', 'MultiLineString']:
                    densities[lt] = clipped.length.sum()
                else:
                    densities[lt] = clipped.area.sum()
        return max(densities, key=densities.get) if any(v > 0 for v in densities.values()) else 'other'

    gdf['land_type'] = gdf['buffer'].apply(get_dominant_type)
    gdf.drop(columns=['buffer'], inplace=True)

    return gdf

def evaluate_land_type_performance(predictions, df_filtered, label_to_cell_id, cell_centers, visuals_dir):
    """
    Calculates and plots model performance for each land type.
    """
    print("\n" + "="*70)
    print("LAND TYPE PERFORMANCE ANALYSIS")
    print("="*70)

    land_type_results = []

    for model_name, data in predictions.items():
        eval_indices = data['indices']
        
        # Create a temporary DataFrame with predictions and land types for the test set
        test_df = df_filtered.loc[eval_indices].copy()
        test_df['y_pred'] = data['pred']
        test_df['y_true'] = data['true']
        
        for land_type in sorted(test_df['land_type'].unique()):
            subset = test_df[test_df['land_type'] == land_type]
            if subset.empty:
                continue

            # Calculate Accuracy
            accuracy = accuracy_score(subset['y_true'], subset['y_pred'])

            # Calculate Mean Error
            true_coords = subset[['Latitude', 'Longitude']].values
            pred_cell_ids = [label_to_cell_id.get(label) for label in subset['y_pred']]
            
            valid_indices = [i for i, cid in enumerate(pred_cell_ids) if cid is not None and cid in cell_centers]
            
            mean_error = np.nan
            if len(valid_indices) > 0:
                pred_coords = np.array([cell_centers[pred_cell_ids[i]] for i in valid_indices])
                distances = haversine_distance(
                    true_coords[valid_indices, 0], true_coords[valid_indices, 1],
                    pred_coords[:, 0], pred_coords[:, 1]
                )
                mean_error = np.mean(distances)

            land_type_results.append({
                'Model': model_name,
                'Land Type': land_type.capitalize(),
                'Accuracy (%)': accuracy * 100,
                'Mean Error (m)': mean_error
            })

    if not land_type_results:
        print("No land type data to evaluate.")
        return

    results_df = pd.DataFrame(land_type_results)
    print(results_df.to_string(index=False))

    # Create plots
    plt.figure(figsize=(18, 8))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    sns.barplot(data=results_df, x='Land Type', y='Accuracy (%)', hue='Model', palette='cividis')
    plt.title('Model Accuracy by Land Type')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Land Type')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Model')

    # Mean Error Plot
    plt.subplot(1, 2, 2)
    sns.barplot(data=results_df, x='Land Type', y='Mean Error (m)', hue='Model', palette='cividis')
    plt.title('Model Mean Error by Land Type')
    plt.ylabel('Mean Error (meters)')
    plt.xlabel('Land Type')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Model')

    plt.tight_layout()
    save_path = os.path.join(visuals_dir, 'land_type_performance_comparison.png')
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"\n✅ Land type performance plot saved to {save_path}")

def evaluate_models(data_path):
    """
    Loads trained models and evaluates their performance with comprehensive metrics.
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

    # Add land type labels to the data
    print("🏷️ Labeling data points by dominant land type...")
    try:
        df_filtered = label_land_type_by_buffer(df_filtered)
        print("✅ Land type labeling complete.")
    except Exception as e:
        print(f"❌ Could not perform land type labeling. Please ensure 'geopandas' is installed and shapefiles are in 'ITUMapData' directory. Error: {e}")
        df_filtered['land_type'] = 'unknown'

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
            
            # Get probabilities if available
            y_pred_proba = None
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test_tree)
            
            predictions[model_name] = {
                'pred': y_pred, 
                'true': y_test_tree, 
                'indices': test_indices_tree,
                'time': pred_time,
                'proba': y_pred_proba
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
            'time': pred_time,
            'proba': y_pred_proba
        }
        print(f"Prediction time: {pred_time:.3f} seconds")

    # Ensemble model (Weighted Voting)
    from collections import defaultdict

    available_models = [name for name in ['XGBoost', 'RandomForest', 'LSTM'] if name in predictions]

    if len(available_models) >= 2:
        print("\n--- Evaluating Ensemble (Weighted Voting) ---")

        common_indices = set(predictions[available_models[0]]['indices'])
        for name in available_models[1:]:
            common_indices &= set(predictions[name]['indices'])

        if common_indices:
            common_indices = sorted(list(common_indices))
            ensemble_preds = []
            ensemble_true = []

            model_weights = {
                'XGBoost': 0.2,
                'RandomForest': 0.3,
                'LSTM': 0.5
            }

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
                ensemble_true.append(model_preds_by_index[available_models[0]][idx])

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

    # Generate comprehensive reports and visualizations
    results_data = []
    comprehensive_results = {}
    label_to_cell_id = {label: cell_id for label, cell_id in enumerate(label_encoder.classes_)}
    
    # Create subplots for error distribution
    fig, axes = plt.subplots(1, len(predictions), figsize=(6*len(predictions), 5), squeeze=False)
    
    for idx, (name, data) in enumerate(predictions.items()):
        ax = axes[0, idx]
        y_pred = data['pred']
        y_true = data['true']
        eval_indices = data['indices']
        pred_time = data['time']
        
        print(f"\n{'='*50}")
        print(f"--- Comprehensive Evaluation Report for {name} ---")
        print(f"{'='*50}")
        
        # Get predicted coordinates
        true_coords = df_filtered.loc[eval_indices][['Latitude', 'Longitude']].values
        pred_cell_ids = [label_to_cell_id.get(label) for label in y_pred]
        
        valid_indices = [i for i, cid in enumerate(pred_cell_ids) if cid is not None and cid in cell_centers]
        if len(valid_indices) > 0:
            pred_coords = np.array([cell_centers[pred_cell_ids[i]] for i in valid_indices])
            
            # 1. Location metrics
            location_metrics, distances = calculate_comprehensive_location_metrics(
                true_coords[valid_indices], pred_coords
            )
            
            # 2. Classification metrics
            class_metrics = calculate_advanced_classification_metrics(
                y_true, y_pred, data.get('proba')
            )
            
            # 3. 5G-specific metrics
            g5_metrics = calculate_5g_specific_metrics(
                df_filtered.loc[eval_indices], y_pred, y_true, label_to_cell_id, cell_centers
            )
            
            # 4. Spatial coherence (if sequential data)
            spatial_metrics = {}
            if 'timestamp' in df_filtered.columns:
                spatial_metrics = calculate_spatial_coherence_metrics(
                    true_coords[valid_indices], pred_coords, 
                    df_filtered.loc[eval_indices[valid_indices]]['timestamp'].values
                )
            
            # 5. Error distribution
            dist_metrics = analyze_error_distribution(distances)
            
            # Combine all metrics
            comprehensive_results[name] = {
                **location_metrics,
                **class_metrics,
                **g5_metrics,
                **spatial_metrics,
                **dist_metrics
            }
            
            # Print key metrics
            print(f"\n📊 LOCATION ACCURACY METRICS:")
            print(f"  Mean Error: {location_metrics['mean_error']:.2f} meters")
            print(f"  Median Error: {location_metrics['median_error']:.2f} meters")
            print(f"  RMSE: {location_metrics['rmse']:.2f} meters")
            print(f"  CEP50: {location_metrics['cep50']:.2f} meters")
            print(f"  CEP95: {location_metrics['cep95']:.2f} meters")
            
            print(f"\n📍 SUCCESS RATES:")
            print(f"  Within 5m: {location_metrics['success_rate_5m']*100:.1f}%")
            print(f"  Within 10m: {location_metrics['success_rate_10m']*100:.1f}%")
            print(f"  Within 15m: {location_metrics['success_rate_15m']*100:.1f}%")
            
            print(f"\n🎯 CLASSIFICATION METRICS:")
            print(f"  Accuracy: {class_metrics.get('accuracy', accuracy_score(y_true, y_pred))*100:.2f}%")
            print(f"  Balanced Accuracy: {class_metrics['balanced_accuracy']*100:.2f}%")
            print(f"  Matthews Correlation: {class_metrics['matthews_correlation']:.3f}")
            print(f"  Cohen's Kappa: {class_metrics['cohen_kappa']:.3f}")
            
            if 'top_3_accuracy' in class_metrics:
                print(f"  Top-3 Accuracy: {class_metrics['top_3_accuracy']*100:.2f}%")
            
            print(f"\n📊 DISTRIBUTION ANALYSIS:")
            print(f"  Best Fit Distribution: {dist_metrics['best_fit_distribution']}")
            print(f"  Skewness: {location_metrics['skewness']:.3f}")
            print(f"  Kurtosis: {location_metrics['kurtosis']:.3f}")
            
            # Plot error distribution for this model
            ax.hist(distances, bins=50, alpha=0.7, color=plt.cm.tab10(idx))
            ax.axvline(location_metrics['mean_error'], color='red', linestyle='--', 
                      label=f'Mean: {location_metrics["mean_error"]:.1f}m')
            ax.axvline(location_metrics['median_error'], color='green', linestyle='--', 
                      label=f'Median: {location_metrics["median_error"]:.1f}m')
            ax.set_xlabel('Error Distance (meters)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{name} Error Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Create comprehensive visualizations
            create_comprehensive_visualizations(name, distances, comprehensive_results[name], visuals_dir)
            
            # Save confusion matrix
            cm_path = os.path.join(visuals_dir, f'{name}_confusion_matrix.png')
            plot_confusion_matrix(y_true, y_pred, name, cm_path)
            
            # Add to results summary
            results_data.append({
                'Model': name, 
                'Accuracy (%)': class_metrics.get('accuracy', accuracy_score(y_true, y_pred)) * 100,
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
    plt.savefig(os.path.join(visuals_dir, 'error_distributions.png'), dpi=300)
    plt.close()

    # Create summary comparison visualizations
    if results_data:
        results_df = pd.DataFrame(results_data).round(2)
        
        # Create metrics comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Accuracy Metrics Comparison
        ax = axes[0, 0]
        acc_metrics = ['Accuracy (%)', 'Balanced Acc (%)', 'Success <10m (%)', 'Success <15m (%)']
        results_melted = results_df.melt(id_vars='Model', value_vars=acc_metrics)
        sns.barplot(data=results_melted, x='variable', y='value', hue='Model', ax=ax)
        ax.set_title('Accuracy Metrics Comparison')
        ax.set_xlabel('')
        ax.set_ylabel('Percentage (%)')
        ax.tick_params(axis='x', rotation=15)
        
        # Plot 2: Error Metrics Comparison
        ax = axes[0, 1]
        error_metrics = ['Mean Error (m)', 'Median Error (m)', 'RMSE (m)']
        results_melted = results_df.melt(id_vars='Model', value_vars=error_metrics)
        sns.barplot(data=results_melted, x='variable', y='value', hue='Model', ax=ax)
        ax.set_title('Error Metrics Comparison')
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
        ax.set_title('Circular Error Probable Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        
        # Plot 4: Model Performance Overview (Radar Chart)
        ax = axes[1, 1]
        categories = ['Accuracy', 'Speed', 'Precision', 'Robustness']
        
        # Normalize metrics for radar chart
        for i, row in results_df.iterrows():
            model = row['Model']
            accuracy = row['Accuracy (%)'] / 100
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
        ax.set_title('Model Performance Overview')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(visuals_dir, 'comprehensive_model_comparison.png'), dpi=300)
        plt.show()

        # Print final summary
        print("\n" + "="*100)
        print("COMPREHENSIVE PERFORMANCE SUMMARY")
        print("="*100)
        print(results_df.to_string(index=False))
        
        # Save comprehensive results
        results_df.to_csv(os.path.join(visuals_dir, 'comprehensive_performance_results.csv'), index=False)
        
        # Save detailed metrics to JSON
        import json
        with open(os.path.join(visuals_dir, 'detailed_metrics.json'), 'w') as f:
            json.dump(comprehensive_results, f, indent=2)
        
        print(f"\n✅ Comprehensive results saved to {visuals_dir}")
        plot_scatter_predictions(df_filtered, predictions, cell_centers, label_to_cell_id, visuals_dir)
        # Identify best models
        best_accuracy_model = results_df.loc[results_df['Accuracy (%)'].idxmax(), 'Model']
        best_error_model = results_df.loc[results_df['Mean Error (m)'].idxmin(), 'Model']
        best_cep50_model = results_df.loc[results_df['CEP50 (m)'].idxmin(), 'Model']
        
        print(f"\n🏆 BEST PERFORMERS:")
        print(f"  Highest Accuracy: {best_accuracy_model}")
        print(f"  Lowest Mean Error: {best_error_model}")
        print(f"  Best CEP50: {best_cep50_model}")

    # Evaluate performance by land type if available
    if 'land_type' in df_filtered.columns and 'unknown' not in df_filtered['land_type'].unique():
        evaluate_land_type_performance(predictions, df_filtered, label_to_cell_id, cell_centers, visuals_dir)

    # Print 5G-specific insights if available
    print("\n" + "="*70)
    print("5G SIGNAL CONDITION INSIGHTS")
    print("="*70)
    
    for model_name, metrics in comprehensive_results.items():
        print(f"\n{model_name}:")
        rsrp_metrics = {k: v for k, v in metrics.items() if 'rsrp' in k.lower()}
        sinr_metrics = {k: v for k, v in metrics.items() if 'sinr' in k.lower()}
        
        if rsrp_metrics:
            print("  RSRP Performance:")
            for k, v in rsrp_metrics.items():
                print(f"    {k}: {v:.2f}")
        
        if sinr_metrics:
            print("  SINR Performance:")
            for k, v in sinr_metrics.items():
                print(f"    {k}: {v:.2f}")

def plot_scatter_predictions(df_filtered, predictions, cell_centers, label_to_cell_id, save_dir):
    """
    Her model için gerçek ve tahmin edilen koordinatları içeren scatter plot çizer.
    """
    for model_name, data in predictions.items():
        y_pred = data['pred']
        indices = data['indices']
        true_coords = df_filtered.loc[indices][['Latitude', 'Longitude']].values
        pred_cell_ids = [label_to_cell_id.get(label) for label in y_pred]
        
        valid_indices = [i for i, cid in enumerate(pred_cell_ids) if cid is not None and cid in cell_centers]
        if not valid_indices:
            continue
        
        true_valid = true_coords[valid_indices]
        pred_coords = np.array([cell_centers[pred_cell_ids[i]] for i in valid_indices])
        
        plt.figure(figsize=(8, 8))
        plt.scatter(true_valid[:, 1], true_valid[:, 0], c='green', s=20, label='True Location', alpha=0.6)
        plt.scatter(pred_coords[:, 1], pred_coords[:, 0], c='red', s=20, label='Predicted Location', alpha=0.6)
        plt.title(f"Prediction vs True Location - {model_name}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{model_name}_scatter_true_vs_pred.png")
        plt.savefig(save_path, dpi=300)
        plt.close()


# Execution
if __name__ == '__main__':
    import time
    DATA_PATH = os.path.join("Data", "Merged_encoded_filled_filtered_pciCleaned_featureEngineered.xlsx")
    evaluate_models(DATA_PATH)