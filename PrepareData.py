import pandas as pd
import numpy as np
import os

def clean_and_preprocess_5g_data(dl_path, ul_path, scanner_path):
    """
    Loads, cleans, preprocesses, and merges 5G network data from Excel files.

    Args:
        dl_path (str): File path for the 5G Downlink data Excel file.
        ul_path (str): File path for the 5G Uplink data Excel file.
        scanner_path (str): File path for the 5G Scanner data Excel file.

    Returns:
        pandas.DataFrame: A cleaned and merged DataFrame ready for analysis,
                          or None if file loading fails.
    """
    # --- 1. Load Data from Excel ---
    try:
        # Read the specific sheet from each Excel file
        sheet_name = 'Series Formatted Data'
        df_dl = pd.read_excel(dl_path, sheet_name=sheet_name)
        df_ul = pd.read_excel(ul_path, sheet_name=sheet_name)
        df_scanner = pd.read_excel(scanner_path, sheet_name=sheet_name)
        print("✅ Data loaded successfully from Excel files.")
    except FileNotFoundError as e:
        print(f"❌ Error loading files: {e}")
        print("Please ensure the file paths are correct and the files exist.")
        return None
    except ValueError as e:
        print(f"❌ Error reading Excel sheets: {e}")
        print(f"Please ensure the sheet named '{sheet_name}' exists in all Excel files.")
        return None


    # --- 2. Initial Cleaning (for each DataFrame) ---
    def initial_clean(df, name):
        """Removes unnamed columns and columns with high missing rates."""
        print(f"🧹 Initial cleaning for {name}...")
        # Drop columns that start with 'Unnamed'
        original_cols = df.shape[1]
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        print(f"   - Removed {original_cols - df.shape[1]} unnamed columns.")
        
        # Drop columns with more than 90% missing values
        original_cols = df.shape[1]
        missing_thresh = len(df) * 0.1 # Keep columns with at least 10% data
        
        # This is the corrected way to avoid the SettingWithCopyWarning
        df = df.dropna(axis=1, thresh=missing_thresh)
        print(f"   - Removed {original_cols - df.shape[1]} columns with >90% missing values.")
        
        return df

    df_dl = initial_clean(df_dl, "5G_DL")
    df_ul = initial_clean(df_ul, "5G_UL")
    df_scanner = initial_clean(df_scanner, "5G_Scanner")

    # --- 3. Correct Data Types ---
    def correct_dtypes(df, name):
        """Converts Time to datetime and handles numeric conversions."""
        print(f"⚙️ Correcting data types for {name}...")
        # Convert Time column, handling potential errors
        if 'Time' in df.columns:
            df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
            df.dropna(subset=['Time'], inplace=True) # Drop rows where time could not be parsed

        # Convert object columns that look numeric
        for col in df.select_dtypes(include=['object']).columns:
            # Skip known non-numeric columns
            if col not in ['Technology_Mode', 'NR_RRC_MsgType', 'NAS_5GS_MM_MessageType']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    df_dl = correct_dtypes(df_dl, "5G_DL")
    df_ul = correct_dtypes(df_ul, "5G_UL")
    df_scanner = correct_dtypes(df_scanner, "5G_Scanner")

    # --- 4. Merge DataFrames ---
    print("🔗 Merging DataFrames...")
    # Sort all dataframes by time to prepare for as-of merge
    df_dl.sort_values('Time', inplace=True)
    df_ul.sort_values('Time', inplace=True)
    df_scanner.sort_values('Time', inplace=True)

    # Merge DL and UL data, keeping DL suffixes for common columns
    df_merged = pd.merge_asof(df_dl, df_ul, on='Time', suffixes=('_DL', '_UL'))
    
    # Merge the result with scanner data
    df_merged = pd.merge_asof(df_merged, df_scanner, on='Time')
    print("   - Merge complete.")

    # --- 5. Handle Missing Data (Post-Merge) ---
    print("💧 Handling final missing data...")
    # Forward-fill location data to propagate last known position
    for col in ['Latitude', 'Longitude', 'Latitude_DL', 'Longitude_DL', 'Latitude_UL', 'Longitude_UL']:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].ffill()

    # Consolidate location columns, preferring the primary ones from the scanner
    df_merged['Latitude'] = df_merged.get('Latitude').fillna(df_merged.get('Latitude_DL')).fillna(df_merged.get('Latitude_UL'))
    df_merged['Longitude'] = df_merged.get('Longitude').fillna(df_merged.get('Longitude_DL')).fillna(df_merged.get('Longitude_UL'))
    
    # Drop rows if location is still unknown after consolidation
    df_merged.dropna(subset=['Latitude', 'Longitude'], inplace=True)
    print(f"   - Rows after dropping missing locations: {len(df_merged)}")

    # Impute remaining columns
    # Numerical with median
    for col in df_merged.select_dtypes(include=np.number).columns:
        if df_merged[col].isnull().any():
            # This is the corrected way to avoid the FutureWarning
            df_merged[col] = df_merged[col].fillna(df_merged[col].median())
            
    # Categorical with mode
    for col in df_merged.select_dtypes(include=['object', 'category']).columns:
        if df_merged[col].isnull().any():
            mode_val = df_merged[col].mode()
            if not mode_val.empty:
                # This is the corrected way to avoid the FutureWarning
                df_merged[col] = df_merged[col].fillna(mode_val[0])

    # --- 6. Final Cleanup ---
    print("✨ Finalizing the dataset...")
    # Drop redundant, now-consolidated, or unhelpful columns
    cols_to_drop = [
        'Latitude_DL', 'Longitude_DL', 'Latitude_UL', 'Longitude_UL',
        'Message_DL', 'Message_UL', 'Message',
        'Distance', 'Distance_DL', 'Distance_UL',
        'GPS_Confidence', 'GPS_Confidence_DL', 'GPS_Confidence_UL'
    ]
    df_merged.drop(columns=[col for col in cols_to_drop if col in df_merged.columns], inplace=True)
    
    # Set time as index
    df_merged.set_index('Time', inplace=True)
    
    print("\n✅ Preprocessing Complete!")
    return df_merged


# --- Execution ---
if __name__ == '__main__':
    # Define file paths according to your local directory structure
    DL_DATA_PATH = "Data/5G_DL.xlsx"
    UL_DATA_PATH = "Data/5G_UL.xlsx"
    SCANNER_DATA_PATH = "Data/5G_Scanner.xlsx"
    
    # Run the cleaning pipeline
    cleaned_df = clean_and_preprocess_5g_data(DL_DATA_PATH, UL_DATA_PATH, SCANNER_DATA_PATH)
    
    if cleaned_df is not None:
        print("\n--- Final DataFrame Info ---")
        cleaned_df.info()
        print("\n--- Final DataFrame Head ---")
        print(cleaned_df.head())
        
        # --- 7. Save Cleaned Data ---
        print("\n💾 Saving cleaned data to Excel...")
        output_dir = "Data/CleanedData"
        output_path = os.path.join(output_dir, "cleaned_data.xlsx")
        
        try:
            # Create the directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save the dataframe to an Excel file
            cleaned_df.to_excel(output_path)
            print(f"✅ Cleaned data successfully saved to {output_path}")
        except Exception as e:
            print(f"❌ Failed to save the file. Error: {e}")
