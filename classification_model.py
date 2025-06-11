import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import time

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
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

def create_grid_and_assign_cells(df, grid_size_meters=15):
    """
    Creates a grid over the area and assigns each data point to a grid cell.
    """
    lat_min, lat_max = df['Latitude'].min(), df['Latitude'].max()
    lon_min, lon_max = df['Longitude'].min(), df['Longitude'].max()
    lat_step = grid_size_meters / 111000 
    lon_step = grid_size_meters / (111000 * np.cos(np.radians(df['Latitude'].mean())))
    lat_bins = np.arange(lat_min, lat_max + lat_step, lat_step)
    lon_bins = np.arange(lon_min, lon_max + lon_step, lon_step)
    df['lat_cell'] = pd.cut(df['Latitude'], bins=lat_bins, labels=False, include_lowest=True)
    df['lon_cell'] = pd.cut(df['Longitude'], bins=lon_bins, labels=False, include_lowest=True)
    df['cell_id'] = df['lat_cell'].astype(str) + '_' + df['lon_cell'].astype(str)
    
    lat_bin_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
    lon_bin_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
    
    cell_centers = {}
    for cell_id in df['cell_id'].unique():
        if pd.notna(cell_id):
            try:
                lat_idx, lon_idx = map(int, cell_id.split('_'))
                if lat_idx < len(lat_bin_centers) and lon_idx < len(lon_bin_centers):
                    cell_centers[cell_id] = (lat_bin_centers[lat_idx], lon_bin_centers[lon_idx])
            except (ValueError, IndexError):
                print(f"Warning: Could not process cell_id '{cell_id}'. Skipping.")
    print(f"✅ Created a grid with {len(cell_centers)} unique cells.")
    return df, cell_centers

def create_sequences(features, targets, time_steps=10):
    """
    Converts time-series data into sequences for LSTM training.
    """
    X_seq, y_seq, target_indices = [], [], []
    for i in range(len(features) - time_steps):
        X_seq.append(features[i:(i + time_steps)])
        y_seq.append(targets[i + time_steps])
        target_indices.append(i + time_steps)
    return np.array(X_seq), np.array(y_seq), np.array(target_indices)

def train_and_evaluate_lstm_classifier(X_scaled, y_encoded, num_classes, df_filtered, label_to_cell_id, cell_centers):
    """
    Handles the specific data prep, training, and evaluation for the LSTM model.
    """
    print("\n--- Preparing Data and Training LSTM ---")
    TIME_STEPS = 10
    X_seq, y_seq, seq_indices = create_sequences(X_scaled, y_encoded, TIME_STEPS)
    y_seq_categorical = to_categorical(y_seq, num_classes=num_classes)

    X_train, X_test, y_train, y_test, _, test_indices_lstm_seq = train_test_split(
        X_seq, y_seq_categorical, seq_indices, test_size=0.2, random_state=42, stratify=y_seq_categorical
    )
    print(f"LSTM data split into {len(X_train)} training and {len(X_test)} testing sequences.")

    model = Sequential([
        LSTM(100, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.3),
        Dense(50, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    callbacks = [EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True)]
    model.fit(X_train, y_train, epochs=150, batch_size=64, validation_split=0.2, callbacks=callbacks, verbose=0)

    y_pred_proba = model.predict(X_test)
    y_pred_encoded = np.argmax(y_pred_proba, axis=1)
    y_test_encoded = np.argmax(y_test, axis=1)
    
    accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
    
    test_indices_lstm = df_filtered.index[test_indices_lstm_seq]
    true_coords = df_filtered.loc[test_indices_lstm][['Latitude', 'Longitude']].values
    pred_cell_ids = [label_to_cell_id.get(label) for label in y_pred_encoded]
    
    valid_indices = [i for i, cid in enumerate(pred_cell_ids) if cid is not None and cid in cell_centers]
    pred_coords = np.array([cell_centers[pred_cell_ids[i]] for i in valid_indices])
    
    distances = haversine_distance(true_coords[valid_indices, 0], true_coords[valid_indices, 1], pred_coords[:, 0], pred_coords[:, 1])
    mean_error = np.mean(distances)

    return model, mean_error, accuracy


def train_and_save_all_models(featured_data_path):
    """
    Main function to run the model bake-off efficiently and save ALL models.
    """
    script_dir = os.path.dirname(os.path.realpath(__file__))
    full_path = os.path.join(script_dir, featured_data_path)
    df = pd.read_excel(full_path)
    print(f"✅ Data loaded successfully from {full_path}")
    
    df_gridded, cell_centers = create_grid_and_assign_cells(df, grid_size_meters=15)
    cell_counts = df_gridded['cell_id'].value_counts()
    cells_to_keep = cell_counts[cell_counts >= 2].index
    df_filtered = df_gridded[df_gridded['cell_id'].isin(cells_to_keep)].copy().reset_index(drop=True)
    print(f"✅ Filtered out sparse cells. Kept {len(df_filtered)} rows from {len(df_gridded)}.")

    y = df_filtered['cell_id']
    X = df_filtered.select_dtypes(include=np.number).drop(columns=['Latitude', 'Longitude', 'lat_cell', 'lon_cell'])

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models_to_run = {
        'XGBoost': (
            xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False, random_state=42, n_jobs=-1),
            {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [5, 7], 'subsample': [0.8], 'colsample_bytree': [0.8]}
        ),
        'RandomForest': (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {'n_estimators': [100, 200], 'max_depth': [20, None], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]}
        )
    }

    results = {}
    
    # --- Split data for training ---
    X_train, X_test, y_train, y_test, _, test_indices = train_test_split(
        X_scaled, y_encoded, df_filtered.index, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"\nData split into {len(X_train)} training and {len(X_test)} testing samples.")

    # --- NEW: Create a smaller subsample for fast hyperparameter tuning ---
    # Use a fraction of the data for the search, e.g., 20,000 samples
    n_samples_for_tuning = min(20000, len(X_train))
    subsample_indices = np.random.choice(len(X_train), n_samples_for_tuning, replace=False)
    X_train_subsample = X_train[subsample_indices]
    y_train_subsample = y_train[subsample_indices]
    print(f"✅ Created a subsample of {n_samples_for_tuning} for fast hyperparameter tuning.")

    for name, (model, params) in models_to_run.items():
        print(f"\n--- Tuning {name} on subsample... ---")
        search = RandomizedSearchCV(model, params, n_iter=8, cv=3, verbose=1, n_jobs=-1, random_state=42)
        # Fit the search on the SMALLER dataset
        search.fit(X_train_subsample, y_train_subsample)
        
        print(f"Best parameters for {name}: {search.best_params_}")
        
        # --- Train the final model on the FULL training set with the best parameters ---
        print(f"--- Training final {name} model on full dataset... ---")
        best_model = search.best_estimator_
        best_model.fit(X_train, y_train) # Retrain on all the data

        results[name] = {'model': best_model}

    # --- Train and save the LSTM model (it's already relatively fast) ---
    '''
    lstm_model, _, _ = train_and_evaluate_lstm_classifier(
        X_scaled, y_encoded, num_classes, df_filtered, 
        {label: cell_id for label, cell_id in enumerate(label_encoder.classes_)}, 
        cell_centers
    )
    results['LSTM'] = {'model': lstm_model}
    '''
    
    # --- Save All Trained Models ---
    models_dir = os.path.join(script_dir, "Models")
    os.makedirs(models_dir, exist_ok=True)
    print("\n💾 Saving all trained models...")
    for name, res in results.items():
        model_to_save = res['model']
        if name == 'LSTM':
            model_to_save.save(os.path.join(models_dir, f'{name}_model.keras'))
        else:
            joblib.dump(model_to_save, os.path.join(models_dir, f'{name}_model.joblib'))
    print(f"✅ All models saved.")
    
    # Save the essential processing objects
    joblib.dump(label_encoder, os.path.join(models_dir, 'label_encoder.joblib'))
    joblib.dump(cell_centers, os.path.join(models_dir, 'cell_centers.joblib'))
    joblib.dump(scaler, os.path.join(models_dir, 'feature_scaler.joblib'))
    print("✅ All processing assets saved successfully.")
    print("\n\nAll models trained and saved. You can now run evaluate.py to see the final results.")


if __name__ == '__main__':
    DATA_PATH = os.path.join("Data", "Merged_encoded_filled_filtered_pciCleaned_featureEngineered.xlsx")
    train_and_save_all_models(DATA_PATH)
