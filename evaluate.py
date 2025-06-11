import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

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
    for i in range(len(features) - time_steps):
        X_seq.append(features[i:(i + time_steps)])
        y_seq.append(targets[i + time_steps])
        target_indices.append(i + time_steps)
    return np.array(X_seq), np.array(y_seq), np.array(target_indices)

def recreate_grid_from_fixed_boundaries(df, grid_size_meters=15):
    """
    Recreates the fixed grid over the entire ITU Ayazağa campus to assign 
    cell_id to the evaluation data. This ensures consistency with the training process.
    """
    # Fixed ITU Campus Boundaries from ITU_SINIRDUVAR_EPSG4326METADATA.txt
    lat_min = 41.098692000  # LOWER RIGHT Y
    lat_max = 41.110922000  # UPPER LEFT Y
    lon_min = 29.014443000  # UPPER LEFT X
    lon_max = 29.037912000  # LOWER RIGHT X

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
    Loads trained models and evaluates their performance side-by-side.
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
        print("✅ All necessary data and assets loaded.")
    except FileNotFoundError as e:
        print(f"❌ Error loading assets: {e}. Please run classification_model.py first to train and save models.")
        return

    # --- 2. Prepare Data ---
    # Recreate the grid using the fixed campus boundaries for consistency
    df_gridded = recreate_grid_from_fixed_boundaries(df, grid_size_meters=15)
    print("✅ Fixed ITU grid system recreated for evaluation.")
    
    # Filter the data to only include cells that the model was trained on
    known_cells = label_encoder.classes_
    df_filtered = df_gridded[df_gridded['cell_id'].isin(known_cells)].copy().reset_index(drop=True)
    
    if df_filtered.empty:
        print("❌ Error: No known cell IDs found in the dataset after creating grid. Cannot evaluate.")
        return
        
    print(f"✅ Found {len(df_filtered)} rows with known cell IDs for evaluation.")

    y = df_filtered['cell_id']
    X = df_filtered.select_dtypes(include=np.number).drop(columns=['Latitude', 'Longitude', 'lat_cell', 'lon_cell'])
    
    X_scaled = scaler.transform(X)
    y_encoded = label_encoder.transform(y)
    
    # --- FIX for ValueError: Stratified split requires at least 2 members per class ---
    # Ensure all classes in the set to be split have at least 2 members for stratification.
    y_series = pd.Series(y_encoded)
    class_counts = y_series.value_counts()
    classes_to_keep = class_counts[class_counts >= 2].index

    # Create a boolean mask for rows with classes that are large enough
    is_class_large_enough = y_series.isin(classes_to_keep)

    # Apply the mask to all relevant data before the split
    df_filtered = df_filtered[is_class_large_enough].reset_index(drop=True)
    X_scaled = X_scaled[is_class_large_enough.values]
    y_encoded = y_encoded[is_class_large_enough.values]

    print(f"✅ Filtered out single-member classes. Kept {len(df_filtered)} rows for splitting.")
    # --- END FIX ---

    # --- 3. Create Consistent Test Sets for All Models ---
    _, X_test_tree, _, y_test_tree, _, test_indices_tree = train_test_split(
        X_scaled, y_encoded, df_filtered.index, test_size=0.2, random_state=42, stratify=y_encoded
    )

    TIME_STEPS = 10
    X_seq, y_seq, seq_indices = create_sequences(X_scaled, y_encoded, TIME_STEPS)
    '''
    _, X_test_lstm, _, y_test_lstm, _, test_indices_lstm_seq = train_test_split(
        X_seq, y_seq, seq_indices, test_size=0.2, random_state=42, stratify=y_seq
    )
    test_indices_lstm = df_filtered.index[test_indices_lstm_seq]
    '''
    # --- 4. Generate Predictions for Each Model ---
    predictions = {}
    
    for model_name in ['XGBoost', 'RandomForest']:
        model_path = os.path.join(models_dir, f'{model_name}_model.joblib')
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            y_pred = model.predict(X_test_tree)
            predictions[model_name] = {'pred': y_pred, 'true': y_test_tree, 'indices': test_indices_tree}

    '''
    lstm_path = os.path.join(models_dir, 'LSTM_model.keras')
    if os.path.exists(lstm_path):
        model = load_model(lstm_path)
        y_pred_proba = model.predict(X_test_lstm)
        y_pred_lstm = np.argmax(y_pred_proba, axis=1)
        predictions['LSTM'] = {'pred': y_pred_lstm, 'true': y_test_lstm, 'indices': test_indices_lstm}
    '''


    # --- 5. Generate Reports and Visualizations ---
    results_data = []
    plt.figure(figsize=(15, 12))
    colors = {'XGBoost': 'red', 'RandomForest': 'green', 'LSTM': 'purple'}

    for name, data in predictions.items():
        y_pred = data['pred']
        y_true = data['true']
        eval_indices = data['indices']
        
        print(f"\n--- Evaluation Report for {name} ---")
        
        # Get the actual class names for the report
        true_labels = np.unique(y_true)
        pred_labels = np.unique(y_pred)
        all_labels = np.unique(np.concatenate((true_labels, pred_labels)))
        target_names = label_encoder.inverse_transform(all_labels)

        report = classification_report(y_true, y_pred, labels=all_labels, target_names=target_names, zero_division=0)
        print(report)
        
        label_to_cell_id = {label: cell_id for label, cell_id in enumerate(label_encoder.classes_)}
        true_coords = df_filtered.loc[eval_indices][['Latitude', 'Longitude']].values
        pred_cell_ids = [label_to_cell_id.get(label) for label in y_pred]
        
        valid_indices = [i for i, cid in enumerate(pred_cell_ids) if cid is not None and cid in cell_centers]
        pred_coords = np.array([cell_centers[pred_cell_ids[i]] for i in valid_indices])
        
        distances = haversine_distance(
            true_coords[valid_indices, 0], true_coords[valid_indices, 1], 
            pred_coords[:, 0], pred_coords[:, 1]
        )
        mean_error = np.mean(distances)
        accuracy = accuracy_score(y_true, y_pred)
        results_data.append({'Model': name, 'Accuracy (%)': accuracy * 100, 'Mean Error (m)': mean_error})

        plt.scatter(pred_coords[:500, 1], pred_coords[:500, 0], color=colors[name], label=f'Predicted Path ({name})', s=25, alpha=0.6)

    true_coords_plot = df_filtered.loc[predictions['XGBoost']['indices']][['Latitude', 'Longitude']].values
    plt.scatter(true_coords_plot[:500, 1], true_coords_plot[:500, 0], color='black', label='Actual Path', s=15, alpha=0.9, zorder=5)

    plt.title('Comparison of Model Predictions vs. Actual Path')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(visuals_dir, 'model_prediction_comparison_map.png'))
    plt.show()

    results_df = pd.DataFrame(results_data).round(2)
    print("\n--- Final Performance Summary ---")
    print(results_df)

    results_df.set_index('Model').plot(kind='bar', subplots=True, layout=(1, 2), figsize=(15, 5), legend=False, rot=0)
    plt.suptitle('Model Performance Comparison')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(visuals_dir, 'model_performance_comparison_chart.png'))
    plt.show()

# --- Execution ---
if __name__ == '__main__':
    DATA_PATH = os.path.join("Data", "Merged_encoded_filled_filtered_pciCleaned_featureEngineered.xlsx")
    evaluate_models(DATA_PATH)
