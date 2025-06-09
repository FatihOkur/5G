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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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
    X_seq, y_seq = [], []
    # We need the original indices to get the true coordinates later
    target_indices = []
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
    
    # 1. Create sequences for LSTM
    TIME_STEPS = 10
    X_seq, y_seq, seq_indices = create_sequences(X_scaled, y_encoded, TIME_STEPS)
    
    # One-hot encode the target variable for classification with Keras
    y_seq_categorical = to_categorical(y_seq, num_classes=num_classes)

    # 2. Split sequence data
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X_seq, y_seq_categorical, seq_indices, test_size=0.2, random_state=42, stratify=y_seq_categorical
    )
    print(f"LSTM data split into {len(X_train)} training and {len(X_test)} testing sequences.")

    # 3. Build and train model
    model = Sequential([
        LSTM(100, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.3),
        Dense(50, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax') # Softmax for multi-class classification
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    callbacks = [EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True)]
    model.fit(X_train, y_train, epochs=150, batch_size=64, validation_split=0.2, callbacks=callbacks, verbose=1)

    # 4. Evaluate
    y_pred_proba = model.predict(X_test)
    y_pred_encoded = np.argmax(y_pred_proba, axis=1)
    y_test_encoded = np.argmax(y_test, axis=1)
    
    accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
    
    # Use the saved sequence indices to get the correct true coordinates
    true_coords = df_filtered.iloc[test_indices][['Latitude', 'Longitude']].values
    pred_cell_ids = [label_to_cell_id.get(label) for label in y_pred_encoded]
    
    valid_indices = [i for i, cid in enumerate(pred_cell_ids) if cid is not None and cid in cell_centers]
    pred_coords = np.array([cell_centers[pred_cell_ids[i]] for i in valid_indices])
    
    distances = haversine_distance(true_coords[valid_indices, 0], true_coords[valid_indices, 1], pred_coords[:, 0], pred_coords[:, 1])
    mean_error = np.mean(distances)

    print(f"✅ LSTM Accuracy: {accuracy * 100:.2f}%")
    print(f"🏆 LSTM Mean Positioning Error: {mean_error:.2f} meters")

    return model, mean_error, accuracy


def train_and_evaluate_models(featured_data_path):
    """
    Main function to run the model bake-off.
    """
    # --- 1. Load and Prepare Data ---
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
    label_to_cell_id = {label: cell_id for label, cell_id in enumerate(label_encoder.classes_)}
    num_classes = len(label_encoder.classes_) # Needed for LSTM output layer

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- 2. Define Tree-Based Models ---
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

    # --- 3. Train and Tune Tree-Based Models ---
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X_scaled, y_encoded, df_filtered.index, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"\nData split for Tree Models into {len(X_train)} training and {len(X_test)} testing samples.")

    for name, (model, params) in models_to_run.items():
        print(f"\n--- Tuning and Training {name} ---")
        search = RandomizedSearchCV(model, params, n_iter=8, cv=3, verbose=1, n_jobs=-1, random_state=42)
        search.fit(X_train, y_train)
        print(f"Best parameters for {name}: {search.best_params_}")
        best_model = search.best_estimator_
        
        y_pred_encoded = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred_encoded)
        
        true_coords = df_filtered.loc[test_indices][['Latitude', 'Longitude']].values
        pred_cell_ids = [label_to_cell_id.get(label) for label in y_pred_encoded]
        valid_indices = [i for i, cid in enumerate(pred_cell_ids) if cid is not None and cid in cell_centers]
        pred_coords = np.array([cell_centers[pred_cell_ids[i]] for i in valid_indices])
        
        distances = haversine_distance(true_coords[valid_indices, 0], true_coords[valid_indices, 1], pred_coords[:, 0], pred_coords[:, 1])
        mean_error = np.mean(distances)

        print(f"✅ {name} Accuracy: {accuracy * 100:.2f}%")
        print(f"🏆 {name} Mean Positioning Error: {mean_error:.2f} meters")
        results[name] = {'model': best_model, 'error': mean_error, 'accuracy': accuracy}

    # --- 4. Train and Evaluate LSTM Model ---
    lstm_model, lstm_error, lstm_accuracy = train_and_evaluate_lstm_classifier(
        X_scaled, y_encoded, num_classes, df_filtered, label_to_cell_id, cell_centers
    )
    results['LSTM'] = {'model': lstm_model, 'error': lstm_error, 'accuracy': lstm_accuracy}


    # --- 5. Select and Save the Best Model ---
    best_model_name = min(results, key=lambda k: results[k]['error'])
    best_model_info = results[best_model_name]
    
    print(f"\n--- 🏆 Overall Best Model: {best_model_name} with {best_model_info['error']:.2f}m error ---")
    
    print("💾 Saving the best model and necessary assets...")
    models_dir = os.path.join(script_dir, "Models")
    os.makedirs(models_dir, exist_ok=True)
    if best_model_name == 'LSTM':
        best_model_info['model'].save(os.path.join(models_dir, 'best_classification_model.keras'))
    else:
        joblib.dump(best_model_info['model'], os.path.join(models_dir, 'best_classification_model.joblib'))
    
    joblib.dump(label_encoder, os.path.join(models_dir, 'label_encoder.joblib'))
    joblib.dump(cell_centers, os.path.join(models_dir, 'cell_centers.joblib'))
    joblib.dump(scaler, os.path.join(models_dir, 'feature_scaler.joblib'))
    print("✅ All champion model assets saved successfully.")

# --- Execution ---
if __name__ == '__main__':
    DATA_PATH = os.path.join("Data", "featured_data_final_v3.xlsx")
    train_and_evaluate_models(DATA_PATH)
