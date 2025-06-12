import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# Set random seeds for reproducibility
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

def create_grid_and_assign_cells(df, grid_size_meters=10):
    """
    Creates a fixed grid over the entire ITU Ayazağa campus and assigns each 
    data point to a grid cell. The grid boundaries are based on the 
    ITU_SINIRDUVAR_EPSG4326 metadata.
    """
    # Fixed ITU Campus Boundaries from ITU_SINIRDUVAR_EPSG4326METADATA.txt
    lat_min = 41.098692000  # LOWER RIGHT Y
    lat_max = 41.110922000  # UPPER LEFT Y
    lon_min = 29.014443000  # UPPER LEFT X
    lon_max = 29.037912000  # LOWER RIGHT X

    # Calculate step size for latitude
    lat_step = grid_size_meters / 111000 

    # Use the mean latitude of the entire campus for a more accurate longitude step
    mean_latitude_campus = (lat_min + lat_max) / 2
    lon_step = grid_size_meters / (111000 * np.cos(np.radians(mean_latitude_campus)))

    # Create the grid bins based on the fixed boundaries
    lat_bins = np.arange(lat_min, lat_max + lat_step, lat_step)
    lon_bins = np.arange(lon_min, lon_max + lon_step, lon_step)

    # Assign each data point in the dataframe to a grid cell
    df['lat_cell'] = pd.cut(df['Latitude'], bins=lat_bins, labels=False, include_lowest=True)
    df['lon_cell'] = pd.cut(df['Longitude'], bins=lon_bins, labels=False, include_lowest=True)

    # Create the unique cell_id string
    df['cell_id'] = df['lat_cell'].astype(str) + '_' + df['lon_cell'].astype(str)
    
    # Create a complete dictionary of all cell centers
    lat_bin_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
    lon_bin_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
    
    cell_centers = {}
    for lat_idx in range(len(lat_bin_centers)):
        for lon_idx in range(len(lon_bin_centers)):
            cell_id = f"{lat_idx}_{lon_idx}"
            cell_centers[cell_id] = (lat_bin_centers[lat_idx], lon_bin_centers[lon_idx])
            
    print(f"✅ Created a fixed ITU grid with {len(cell_centers)} unique cells.")
    
    return df, cell_centers

def create_sequences(features, targets, time_steps=10):
    """
    Converts time-series data into sequences for LSTM training.
    Enhanced with padding and better sequence handling.
    """
    X_seq, y_seq, target_indices = [], [], []
    
    # Pad the beginning with zeros for initial sequences
    padded_features = np.vstack([np.zeros((time_steps-1, features.shape[1])), features])
    
    for i in range(len(targets)):
        # Get sequence from padded features
        X_seq.append(padded_features[i:i+time_steps])
        y_seq.append(targets[i])
        target_indices.append(i)
    
    return np.array(X_seq), np.array(y_seq), np.array(target_indices)

def build_lstm_model(input_shape, num_classes):
    """
    Builds an enhanced LSTM model for cell classification.
    Uses bidirectional LSTM with batch normalization and dropout.
    """
    model = Sequential([
        # First Bidirectional LSTM layer
        Bidirectional(LSTM(128, return_sequences=True, 
                          kernel_regularizer=tf.keras.regularizers.l2(0.01)),
                     input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        
        # Second Bidirectional LSTM layer
        Bidirectional(LSTM(64, return_sequences=False,
                          kernel_regularizer=tf.keras.regularizers.l2(0.01))),
        BatchNormalization(),
        Dropout(0.3),
        
        # Dense layers
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_and_evaluate_lstm_classifier(X_scaled, y_encoded, num_classes, df_filtered, label_to_cell_id, cell_centers):
    """
    Handles the specific data prep, training, and evaluation for the LSTM model.
    Enhanced with better preprocessing and model architecture.
    """
    print("\n--- Preparing Data and Training LSTM ---")
    TIME_STEPS = 10
    
    # Create sequences
    X_seq, y_seq, seq_indices = create_sequences(X_scaled, y_encoded, TIME_STEPS)
    
    # Convert to categorical
    y_seq_categorical = to_categorical(y_seq, num_classes=num_classes)
    
    # Split data with stratification
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X_seq, y_seq_categorical, seq_indices, 
        test_size=0.2, random_state=42, stratify=y_seq
    )
    
    print(f"LSTM data split into {len(X_train)} training and {lhissen(X_test)} testing sequences.")
    print(f"Input shape: {X_train.shape}, Output shape: {y_train.shape}")
    
    # Build model
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]), num_classes)
    
    # Compile with custom learning rate
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)
    ]
    
    # Train model
    print("Training LSTM model...")
    history = model.fit(
        X_train, y_train, 
        epochs=100, 
        batch_size=64, 
        validation_split=0.2, 
        callbacks=callbacks, 
        verbose=1
    )
    
    # Evaluate
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred_encoded = np.argmax(y_pred_proba, axis=1)
    y_test_encoded = np.argmax(y_test, axis=1)
    
    accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
    print(f"LSTM Test Accuracy: {accuracy:.4f}")
    
    # Calculate location errors
    test_indices_lstm = df_filtered.index[test_idx]
    true_coords = df_filtered.loc[test_indices_lstm][['Latitude', 'Longitude']].values
    pred_cell_ids = [label_to_cell_id.get(label) for label in y_pred_encoded]
    
    valid_indices = [i for i, cid in enumerate(pred_cell_ids) if cid is not None and cid in cell_centers]
    pred_coords = np.array([cell_centers[pred_cell_ids[i]] for i in valid_indices])
    
    if len(valid_indices) > 0:
        distances = haversine_distance(
            true_coords[valid_indices, 0], true_coords[valid_indices, 1], 
            pred_coords[:, 0], pred_coords[:, 1]
        )
        mean_error = np.mean(distances)
        print(f"LSTM Mean Location Error: {mean_error:.2f} meters")
    else:
        mean_error = np.inf
        print("Warning: No valid predictions for error calculation")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('LSTM Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('LSTM Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('lstm_training_history.png')
    plt.close()

    return model, mean_error, accuracy

def train_and_save_all_models(featured_data_path):
    """
    Main function to run the model bake-off efficiently and save ALL models including LSTM.
    """
    script_dir = os.path.dirname(os.path.realpath(__file__))
    full_path = os.path.join(script_dir, featured_data_path)
    df = pd.read_excel(full_path)
    print(f"✅ Data loaded successfully from {full_path}")
    
    # Create grid and filter sparse cells
    df_gridded, cell_centers = create_grid_and_assign_cells(df, grid_size_meters=10)
    cell_counts = df_gridded['cell_id'].value_counts()
    cells_to_keep = cell_counts[cell_counts >= 5].index  # Increased threshold for LSTM
    df_filtered = df_gridded[df_gridded['cell_id'].isin(cells_to_keep)].copy().reset_index(drop=True)
    print(f"✅ Filtered out sparse cells. Kept {len(df_filtered)} rows from {len(df_gridded)}.")

    # Prepare features and targets
    y = df_filtered['cell_id']
    X = df_filtered.select_dtypes(include=np.number).drop(columns=['Latitude', 'Longitude', 'lat_cell', 'lon_cell'])

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    print(f"Number of classes: {num_classes}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Tree-based models configuration
    models_to_run = {
        'XGBoost': (
            xgb.XGBClassifier(
                objective='multi:softprob', 
                eval_metric='mlogloss', 
                use_label_encoder=False, 
                random_state=42, 
                n_jobs=-1
            ),
            {
                'n_estimators': [100, 200, 300], 
                'learning_rate': [0.05, 0.1, 0.15], 
                'max_depth': [5, 7, 9], 
                'subsample': [0.7, 0.8, 0.9], 
                'colsample_bytree': [0.7, 0.8, 0.9]
            }
        ),
        'RandomForest': (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {
                'n_estimators': [100, 200, 300], 
                'max_depth': [15, 20, 25, None], 
                'min_samples_split': [2, 5, 10], 
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
        )
    }

    results = {}
    
    # Split data for tree-based models
    X_train, X_test, y_train, y_test, _, test_indices = train_test_split(
        X_scaled, y_encoded, df_filtered.index, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"\nData split into {len(X_train)} training and {len(X_test)} testing samples.")

    # Create subsample for hyperparameter tuning
    n_samples_for_tuning = min(20000, len(X_train))
    subsample_indices = np.random.choice(len(X_train), n_samples_for_tuning, replace=False)
    X_train_subsample = X_train[subsample_indices]
    y_train_subsample = y_train[subsample_indices]
    print(f"✅ Created a subsample of {n_samples_for_tuning} for fast hyperparameter tuning.")

    # Train tree-based models
    for name, (model, params) in models_to_run.items():
        print(f"\n--- Tuning {name} on subsample... ---")
        search = RandomizedSearchCV(
            model, params, n_iter=10, cv=3, 
            verbose=1, n_jobs=-1, random_state=42, scoring='f1_weighted'
        )
        search.fit(X_train_subsample, y_train_subsample)
        
        print(f"Best parameters for {name}: {search.best_params_}")
        
        # Train final model on full dataset
        print(f"--- Training final {name} model on full dataset... ---")
        best_model = search.best_estimator_
        start_time = time.time()
        best_model.fit(X_train, y_train)
        training_time = time.time() - start_time
        print(f"{name} training completed in {training_time:.2f} seconds")
        
        # Evaluate
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} Test Accuracy: {accuracy:.4f}")
        
        results[name] = {'model': best_model, 'accuracy': accuracy}

    # Train LSTM model
    label_to_cell_id = {label: cell_id for label, cell_id in enumerate(label_encoder.classes_)}
    lstm_model, lstm_error, lstm_accuracy = train_and_evaluate_lstm_classifier(
        X_scaled, y_encoded, num_classes, df_filtered, 
        label_to_cell_id, cell_centers
    )
    results['LSTM'] = {'model': lstm_model, 'accuracy': lstm_accuracy, 'error': lstm_error}

    # Save all models
    models_dir = os.path.join(script_dir, "Models")
    os.makedirs(models_dir, exist_ok=True)
    print("\n💾 Saving all trained models...")
    
    for name, res in results.items():
        model_to_save = res['model']
        if name == 'LSTM':
            model_to_save.save(os.path.join(models_dir, f'{name}_model.keras'))
        else:
            joblib.dump(model_to_save, os.path.join(models_dir, f'{name}_model.joblib'))
        print(f"✅ {name} model saved - Accuracy: {res['accuracy']:.4f}")
    
    # Save processing objects
    joblib.dump(label_encoder, os.path.join(models_dir, 'label_encoder.joblib'))
    joblib.dump(cell_centers, os.path.join(models_dir, 'cell_centers.joblib'))
    joblib.dump(scaler, os.path.join(models_dir, 'feature_scaler.joblib'))
    
    # Save feature names for consistency
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, os.path.join(models_dir, 'feature_names.joblib'))
    
    print("✅ All processing assets saved successfully.")
    print("\n📊 Model Performance Summary:")
    for name, res in results.items():
        print(f"{name}: Accuracy = {res['accuracy']:.4f}", end="")
        if 'error' in res:
            print(f", Mean Error = {res['error']:.2f}m")
        else:
            print()
    
    print("\n\n✨ All models trained and saved. You can now run evaluate.py to see the final results.")


if __name__ == '__main__':
    DATA_PATH = os.path.join("Data", "Merged_encoded_filled_filtered_pciCleaned_featureEngineered.xlsx")
    train_and_save_all_models(DATA_PATH)
