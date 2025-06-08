import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda_and_feature_engineering(cleaned_data_path):
    """
    Loads cleaned data, performs EDA, engineers new features, and saves the result.

    Args:
        cleaned_data_path (str): The file path for the cleaned data Excel file.

    Returns:
        pandas.DataFrame: The DataFrame with new features, or None if loading fails.
    """
    # --- 1. Load Cleaned Data ---
    try:
        df = pd.read_excel(cleaned_data_path, index_col='Time')
        print("✅ Cleaned data loaded successfully.")
    except FileNotFoundError:
        print(f"❌ Error: The file was not found at {cleaned_data_path}")
        print("Please ensure the data cleaning script has been run successfully.")
        return None

    # --- 2. Feature Engineering ---
    print("🛠️  Performing Feature Engineering...")
    
    # Calculate signal strength differences between the serving cell (0) and neighbor cells (1, 2)
    df['RSRP_diff_0_1'] = df['NR_Scan_SSB_RSRP_SortedBy_RSRP_0'] - df['NR_Scan_SSB_RSRP_SortedBy_RSRP_1']
    df['RSRQ_diff_0_1'] = df['NR_Scan_SSB_RSRQ_SortedBy_RSRP_0'] - df['NR_Scan_SSB_RSRQ_SortedBy_RSRP_1']
    df['SINR_diff_0_1'] = df['NR_Scan_SSB_SINR_SortedBy_RSRP_0'] - df['NR_Scan_SSB_SINR_SortedBy_RSRP_1']
    df['RSRP_diff_0_2'] = df['NR_Scan_SSB_RSRP_SortedBy_RSRP_0'] - df['NR_Scan_SSB_RSRP_SortedBy_RSRP_2']

    # Calculate rolling statistics to smooth out data and capture trends
    window_size = '5s'
    df['RSRP_0_roll_mean'] = df['NR_Scan_SSB_RSRP_SortedBy_RSRP_0'].rolling(window_size).mean()
    df['SINR_0_roll_mean'] = df['NR_Scan_SSB_SINR_SortedBy_RSRP_0'].rolling(window_size).mean()

    # Drop rows with NaNs created by the rolling window
    df.dropna(inplace=True)
    
    print("   - New features created: Signal differences and rolling means.")
    
    # --- 3. Visualization ---
    print("📊 Generating and saving visualizations...")

    # Create visuals directory if it doesn't exist
    visuals_dir = "Visuals"
    os.makedirs(visuals_dir, exist_ok=True)

    # a) Geographical Path Visualization (colored by primary RSRP)
    plt.figure(figsize=(10, 10))
    sns.scatterplot(
        x='Longitude', 
        y='Latitude', 
        data=df, 
        hue='NR_Scan_SSB_RSRP_SortedBy_RSRP_0', 
        palette='viridis', 
        s=10,
        legend='auto'
    )
    plt.title('Drive Test Path (Colored by Serving Cell RSRP)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.savefig(os.path.join(visuals_dir, 'drive_test_path_by_rsrp.png'))
    plt.show()

    # b) Distribution of New Difference Features
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    sns.histplot(df['RSRP_diff_0_1'], bins=50, ax=axes[0], kde=True, color='skyblue')
    axes[0].set_title('Distribution of RSRP Difference (Cell 0 - Cell 1)')
    sns.histplot(df['RSRQ_diff_0_1'], bins=50, ax=axes[1], kde=True, color='salmon')
    axes[1].set_title('Distribution of RSRQ Difference (Cell 0 - Cell 1)')
    sns.histplot(df['SINR_diff_0_1'], bins=50, ax=axes[2], kde=True, color='lightgreen')
    axes[2].set_title('Distribution of SINR Difference (Cell 0 - Cell 1)')
    plt.tight_layout()
    plt.savefig(os.path.join(visuals_dir, 'signal_difference_distributions.png'))
    plt.show()

    # --- 4. Correlation Analysis ---
    print("📈 Performing and saving Correlation Analysis...")
    plt.figure(figsize=(15, 12))
    
    # Select key numeric features for the heatmap, including new ones
    corr_cols = [
        'Longitude', 'Latitude', 
        'NR_Scan_SSB_RSRP_SortedBy_RSRP_0', 'NR_Scan_SSB_RSRQ_SortedBy_RSRP_0', 'NR_Scan_SSB_SINR_SortedBy_RSRP_0',
        'RSRP_diff_0_1', 'RSRQ_diff_0_1', 'SINR_diff_0_1', 'RSRP_diff_0_2',
        'RSRP_0_roll_mean', 'SINR_0_roll_mean'
    ]
    correlation_matrix = df[corr_cols].corr()

    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Key Features and Location')
    plt.savefig(os.path.join(visuals_dir, 'feature_correlation_heatmap.png'))
    plt.show()
    
    return df

# --- Execution ---
if __name__ == '__main__':
    # Define the path for the cleaned data, now directly in Data/
    CLEANED_DATA_PATH = os.path.join("Data", "cleaned_data.xlsx")
    
    # Run the EDA and feature engineering pipeline
    featured_df = perform_eda_and_feature_engineering(CLEANED_DATA_PATH)
    
    if featured_df is not None:
        print("\n--- DataFrame with New Features ---")
        print(featured_df.head())
        
        # --- 5. Save Featured Data ---
        print("\n💾 Saving data with new features to Excel...")
        # Define output path, now directly in Data/
        output_path = os.path.join("Data", "featured_data.xlsx")
        
        try:
            # Save the dataframe to an Excel file
            featured_df.to_excel(output_path)
            print(f"✅ Featured data successfully saved to {output_path}")
        except Exception as e:
            print(f"❌ Failed to save the file. Error: {e}")
