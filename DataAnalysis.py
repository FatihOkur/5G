import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_signal_features(cleaned_data_path):
    """
    Loads cleaned data and engineers features based ONLY on the available signal data,
    making it suitable for the competition (no data leakage).

    Args:
        cleaned_data_path (str): The file path for the cleaned data Excel file.

    Returns:
        pandas.DataFrame: The DataFrame with new signal-based features.
    """
    # --- 1. Load Data ---
    try:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        full_path = os.path.join(script_dir, cleaned_data_path)
        df = pd.read_excel(full_path)
        print("✅ Cleaned data loaded successfully.")
    except FileNotFoundError as e:
        print(f"❌ Error: A file was not found. {e}")
        return None

    # --- 2. Signal-Based Feature Engineering ---
    # This section creates new features using only the relationships
    # between the different signal measurements.
    print("🛠️  Performing Signal-Only Feature Engineering...")

    # Set Time as index for time-based operations
    df['Time'] = pd.to_datetime(df['Time'])
    df.set_index('Time', inplace=True)
    df.sort_index(inplace=True)

    # Create features based on the difference between the strongest signals
    df['RSRP_diff_0_1'] = df['NR_Scan_SSB_RSRP_SortedBy_RSRP_0'] - df['NR_Scan_SSB_RSRP_SortedBy_RSRP_1']
    df['RSRQ_diff_0_1'] = df['NR_Scan_SSB_RSRQ_SortedBy_RSRP_0'] - df['NR_Scan_SSB_RSRQ_SortedBy_RSRP_1']
    df['SINR_diff_0_1'] = df['NR_Scan_SSB_SINR_SortedBy_RSRP_0'] - df['NR_Scan_SSB_SINR_SortedBy_RSRP_1']
    
    df['RSRP_diff_0_2'] = df['NR_Scan_SSB_RSRP_SortedBy_RSRP_0'] - df['NR_Scan_SSB_RSRP_SortedBy_RSRP_2']
    df['RSRP_diff_1_2'] = df['NR_Scan_SSB_RSRP_SortedBy_RSRP_1'] - df['NR_Scan_SSB_RSRP_SortedBy_RSRP_2']
    
    # Create features based on rolling window statistics to capture trends
    window_size = '5s'
    df['RSRP_0_roll_mean'] = df['NR_Scan_SSB_RSRP_SortedBy_RSRP_0'].rolling(window_size).mean()
    df['SINR_0_roll_mean'] = df['NR_Scan_SSB_SINR_SortedBy_RSRP_0'].rolling(window_size).mean()
    df['RSRP_0_roll_std'] = df['NR_Scan_SSB_RSRP_SortedBy_RSRP_0'].rolling(window_size).std()
    df['SINR_0_roll_std'] = df['NR_Scan_SSB_SINR_SortedBy_RSRP_0'].rolling(window_size).std()

    # Drop rows with NaNs created by the rolling window
    df.dropna(inplace=True)
    print("✅ Created signal difference and rolling window features.")

    # --- 3. Visualization and Correlation ---
    # This helps understand the relationships between the new signal features and location
    print("📊📈 Performing Visualization and Correlation Analysis...")
    visuals_dir = os.path.join(script_dir, "Visuals")
    os.makedirs(visuals_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 12))
    corr_cols = [
        'Longitude', 'Latitude', 
        'NR_Scan_SSB_RSRP_SortedBy_RSRP_0', 'NR_Scan_SSB_RSRQ_SortedBy_RSRP_0', 'NR_Scan_SSB_SINR_SortedBy_RSRP_0',
        'RSRP_diff_0_1', 'SINR_diff_0_1', 'RSRP_0_roll_mean', 'RSRP_0_roll_std'
    ]
    corr_cols_exist = [col for col in corr_cols if col in df.columns]
    correlation_matrix = df[corr_cols_exist].corr()

    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Competition-Safe Signal Features')
    plt.tight_layout()
    plt.savefig(os.path.join(visuals_dir, 'feature_correlation_heatmap_competition.png'))
    plt.show()
    
    return df

# --- Main Execution Block ---
if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    
    # The only input needed is the cleaned data file
    CLEANED_DATA_PATH = os.path.join("Data", "cleaned_data.xlsx")
    
    # --- Run the feature engineering pipeline ---
    featured_df = create_signal_features(CLEANED_DATA_PATH)
    
    if featured_df is not None:
        print("\n--- DataFrame with Competition-Safe Features (Sample) ---")
        print(featured_df.reset_index().head())
        
        # --- Save the new, competition-safe dataset ---
        output_path = os.path.join(script_dir, "Data", "featured_data_competition.xlsx")
        print(f"\n💾 Saving competition-safe data to {output_path}...")
        
        try:
            featured_df.to_excel(output_path)
            print(f"✅ Competition-safe featured data successfully saved!")
        except Exception as e:
            print(f"❌ Failed to save the file. Error: {e}")
