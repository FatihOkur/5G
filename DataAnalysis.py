import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance in meters between two points 
    on the earth (specified in decimal degrees).
    """
    # Convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers.
    return c * r * 1000

def perform_geometric_feature_engineering(cleaned_data_path, cell_info_path):
    """
    Loads cleaned data, performs EDA, engineers new geometric and signal features, 
    and saves the result.

    Args:
        cleaned_data_path (str): The file path for the cleaned data Excel file.
        cell_info_path (str): The file path for the cell info Excel file.

    Returns:
        pandas.DataFrame: The DataFrame with new features, or None if loading fails.
    """
    # --- 1. Load Data ---
    try:
        # Load data, but keep 'Time' as a column for now
        df = pd.read_excel(cleaned_data_path) 
        df_cells = pd.read_excel(cell_info_path, sheet_name='Hücre tablosu')
        print("✅ Cleaned data and cell info loaded successfully.")
    except FileNotFoundError as e:
        print(f"❌ Error: A file was not found. {e}")
        return None
    except ValueError as e:
        print(f"❌ Error reading Excel sheet: {e}. Ensure the sheet 'Hücre tablosu' exists.")
        return None

    # --- 2. Geometric Feature Engineering ---
    print("🛠️  Performing Geometric Feature Engineering...")
    
    # --- FIX: Dynamically find the PCI column ---
    print(f"Columns in cell info file: {df_cells.columns.tolist()}")
    pci_col_name = None
    for col in df_cells.columns:
        if 'pci' in str(col).lower():
            pci_col_name = col
            break

    if pci_col_name is None:
        print("❌ FATAL ERROR: Could not find a 'PCI' column in the cell info file.")
        return None
        
    print(f"   - Found PCI column as: '{pci_col_name}'. Standardizing to 'PCI'.")
    
    # Prepare cell info dataframe with standardized column names
    df_cells_prepared = df_cells[[pci_col_name, 'Latitude', 'Longitude']].copy()
    df_cells_prepared.rename(columns={
        pci_col_name: 'PCI', 
        'Latitude': 'Cell_Latitude', 
        'Longitude': 'Cell_Longitude'
    }, inplace=True)
    df_cells_prepared.drop_duplicates(subset=['PCI'], inplace=True)

    # Merge cell locations and calculate distances for top 4 PCIs
    for i in range(4):
        pci_col = f'NR_Scan_PCI_SortedBy_RSRP_{i}'
        
        # Merge to get cell coordinates
        df = pd.merge(df, df_cells_prepared, left_on=pci_col, right_on='PCI', how='left')
        
        # Calculate distance to cell
        df[f'distance_to_cell_{i}'] = haversine_distance(
            df['Latitude'], df['Longitude'],
            df[f'Cell_Latitude'], df[f'Cell_Longitude']
        )
        
        # Drop the temporary merged columns to keep the dataframe clean
        df.drop(columns=['PCI', 'Cell_Latitude', 'Cell_Longitude'], inplace=True)

    print("   - Created distance features to the 4 nearest cell towers.")

    # --- 3. Signal Feature Engineering ---
    print("🛠️  Performing Signal Feature Engineering...")

    # --- FIX: Set DatetimeIndex before performing time-based rolling window ---
    df['Time'] = pd.to_datetime(df['Time'])
    df.set_index('Time', inplace=True)
    df.sort_index(inplace=True) # Sorting is crucial for rolling operations

    df['RSRP_diff_0_1'] = df['NR_Scan_SSB_RSRP_SortedBy_RSRP_0'] - df['NR_Scan_SSB_RSRP_SortedBy_RSRP_1']
    df['RSRQ_diff_0_1'] = df['NR_Scan_SSB_RSRQ_SortedBy_RSRP_0'] - df['NR_Scan_SSB_RSRQ_SortedBy_RSRP_1']
    df['SINR_diff_0_1'] = df['NR_Scan_SSB_SINR_SortedBy_RSRP_0'] - df['NR_Scan_SSB_SINR_SortedBy_RSRP_1']
    df['RSRP_diff_0_2'] = df['NR_Scan_SSB_RSRP_SortedBy_RSRP_0'] - df['NR_Scan_SSB_RSRP_SortedBy_RSRP_2']
    
    window_size = '5s'
    df['RSRP_0_roll_mean'] = df['NR_Scan_SSB_RSRP_SortedBy_RSRP_0'].rolling(window_size).mean()
    df['SINR_0_roll_mean'] = df['NR_Scan_SSB_SINR_SortedBy_RSRP_0'].rolling(window_size).mean()
    
    # Drop rows with NaNs created by the merge or rolling window
    df.dropna(inplace=True)
    print("   - Created signal difference and rolling mean features.")

    # --- 4. Visualization and Correlation ---
    print("📊📈 Performing Visualization and Correlation Analysis...")
    visuals_dir = "Visuals"
    os.makedirs(visuals_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 12))
    corr_cols = [
        'Longitude', 'Latitude', 
        'distance_to_cell_0', 'distance_to_cell_1', 'distance_to_cell_2', 'distance_to_cell_3',
        'NR_Scan_SSB_RSRP_SortedBy_RSRP_0', 'NR_Scan_SSB_RSRQ_SortedBy_RSRP_0', 'NR_Scan_SSB_SINR_SortedBy_RSRP_0',
        'RSRP_diff_0_1', 'RSRQ_diff_0_1', 'SINR_diff_0_1',
    ]
    corr_cols_exist = [col for col in corr_cols if col in df.columns]
    correlation_matrix = df[corr_cols_exist].corr()

    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix with Geometric Features')
    plt.savefig(os.path.join(visuals_dir, 'feature_correlation_heatmap_geometric.png'))
    plt.show()
    
    return df

# --- Execution ---
if __name__ == '__main__':
    CLEANED_DATA_PATH = os.path.join("Data", "cleaned_data.xlsx")
    CELL_INFO_PATH = os.path.join("BaseStationConfigurationData", "İTÜ 5G Hücre Bilgileri.xlsx")
    
    featured_df = perform_geometric_feature_engineering(CLEANED_DATA_PATH, CELL_INFO_PATH)
    
    if featured_df is not None:
        print("\n--- DataFrame with New Geometric Features ---")
        # Reset index to show time as a column in head()
        print(featured_df.reset_index().head())
        
        print("\n💾 Saving data with new features to Excel...")
        output_path = os.path.join("Data", "featured_data_geometric.xlsx")
        
        try:
            # When saving, the index is automatically written, so no need to reset
            featured_df.to_excel(output_path)
            print(f"✅ Featured data successfully saved to {output_path}")
        except Exception as e:
            print(f"❌ Failed to save the file. Error: {e}")
