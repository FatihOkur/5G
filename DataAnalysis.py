import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import geopandas as gpd
from shapely.geometry import Point, LineString
import rasterio

# --- Helper Functions ---

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

def is_line_of_sight_obstructed(device_point, cell_point, obstacles_gdf):
    """
    Checks if the line between a device and a cell tower is blocked by an obstacle.
    """
    if not device_point or not cell_point or device_point.is_empty or cell_point.is_empty:
        return False
    line_of_sight = LineString([device_point, cell_point])
    return obstacles_gdf.intersects(line_of_sight).any()

def get_elevation_at_point(lon, lat, raster_dataset):
    """
    Gets the elevation from a raster file for a given latitude and longitude.
    """
    if pd.isna(lon) or pd.isna(lat):
        return np.nan
    try:
        # Get the pixel value for the given coordinates
        for val in raster_dataset.sample([(lon, lat)]):
            return val[0]
    except IndexError:
        # This happens if the point is outside the raster's bounds
        return np.nan

def perform_advanced_feature_engineering(cleaned_data_path, cell_info_path, buildings_shapefile_path, vegetation_shapefile_path, elevation_raster_path):
    """
    Loads cleaned data and engineers geometric, obstruction, interaction, and NEW elevation features.
    """
    # --- 1. Load All Required Data ---
    try:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        df = pd.read_excel(os.path.join(script_dir, cleaned_data_path)) 
        df_cells = pd.read_excel(os.path.join(script_dir, cell_info_path), sheet_name='Hücre tablosu')
        print("✅ Cleaned data and cell info loaded successfully.")
        
        buildings_gdf = gpd.read_file(os.path.join(script_dir, buildings_shapefile_path))
        vegetation_gdf = gpd.read_file(os.path.join(script_dir, vegetation_shapefile_path))
        print("✅ Geospatial data for buildings and vegetation loaded.")

        # --- NEW: Load Elevation Data ---
        elevation_raster = rasterio.open(os.path.join(script_dir, elevation_raster_path))
        print("✅ Elevation raster data loaded successfully.")

    except Exception as e:
        print(f"❌ An error occurred during file loading: {e}")
        return None

    # --- 2. Prepare Cell Info & Elevation Data ---
    print("🛠️  Preparing Cell & Elevation Information...")
    pci_col_name = next((col for col in df_cells.columns if 'pci' in str(col).lower()), None)
    if not pci_col_name:
        print("❌ FATAL ERROR: Could not find a 'PCI' column in the cell info file.")
        return None
    
    df_cells_prepared = df_cells[[pci_col_name, 'Latitude', 'Longitude']].copy()
    df_cells_prepared.rename(columns={pci_col_name: 'PCI', 'Latitude': 'Cell_Latitude', 'Longitude': 'Cell_Longitude'}, inplace=True)
    df_cells_prepared.drop_duplicates(subset=['PCI'], inplace=True)

    # --- NEW: Get elevation for each cell tower ---
    df_cells_prepared['cell_elevation'] = df_cells_prepared.apply(
        lambda row: get_elevation_at_point(row['Cell_Longitude'], row['Cell_Latitude'], elevation_raster), axis=1
    )

    # --- NEW: Get elevation for each device location ---
    df['device_elevation'] = df.apply(
        lambda row: get_elevation_at_point(row['Longitude'], row['Latitude'], elevation_raster), axis=1
    )
    
    # --- 3. Full Feature Engineering ---
    print("🛠️  Performing Final Feature Engineering...")
    for i in range(4):
        pci_col = f'NR_Scan_PCI_SortedBy_RSRP_{i}'
        rsrp_col = f'NR_Scan_SSB_RSRP_SortedBy_RSRP_{i}'
        sinr_col = f'NR_Scan_SSB_SINR_SortedBy_RSRP_{i}'
        print(f"   - Processing features for cell tower set {i}...")
        
        df = pd.merge(df, df_cells_prepared, left_on=pci_col, right_on='PCI', how='left')
        
        # Distance Feature
        df[f'distance_to_cell_{i}'] = haversine_distance(df['Latitude'], df['Longitude'], df['Cell_Latitude'], df['Cell_Longitude'])
        
        # Obstruction & Interaction Features
        device_points = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
        cell_points = df.apply(lambda row: Point(row['Cell_Longitude'], row['Cell_Latitude']) if pd.notna(row['Cell_Longitude']) else None, axis=1)
        df[f'building_obstructs_cell_{i}'] = [is_line_of_sight_obstructed(dev_p, cel_p, buildings_gdf) for dev_p, cel_p in zip(device_points, cell_points)]
        df[f'building_obstructs_cell_{i}'] = df[f'building_obstructs_cell_{i}'].astype(int)
        df[f'RSRP_x_building_{i}'] = df[rsrp_col] * df[f'building_obstructs_cell_{i}']
        
        # --- NEW: Elevation Difference Feature ---
        df[f'elevation_diff_cell_{i}'] = df['device_elevation'] - df['cell_elevation']

        df.drop(columns=['PCI', 'Cell_Latitude', 'Cell_Longitude', 'cell_elevation'], inplace=True, errors='ignore')

    print("✅ Created all features including NEW elevation difference.")

    # --- 4. Final Cleanup & Save ---
    df.set_index('Time', inplace=True)
    df.sort_index(inplace=True)
    df.dropna(inplace=True)
    print("✅ Final data cleaned and prepared.")
    
    return df

# --- Main Execution Block ---
if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    
    CLEANED_DATA_PATH = os.path.join("Data", "cleaned_data.xlsx")
    CELL_INFO_PATH = os.path.join("BaseStationConfigurationData", "İTÜ 5G Hücre Bilgileri.xlsx")
    BUILDINGS_SHAPEFILE = os.path.join("ITUMapData", "ITU_3DBINA_EPSG4326.shp")
    VEGETATION_SHAPEFILE = os.path.join("ITUMapData", "ITU_3DVEGETATION_EPSG4326.shp")
    # --- CORRECTED: Path to elevation data now points to the .asc file ---
    ELEVATION_RASTER = os.path.join("ITUMapData", "ITU_YUKSEKLIK_UTM35NWGS84.asc")
    
    featured_df = perform_advanced_feature_engineering(
        CLEANED_DATA_PATH, 
        CELL_INFO_PATH, 
        BUILDINGS_SHAPEFILE, 
        VEGETATION_SHAPEFILE,
        ELEVATION_RASTER
    )
    
    if featured_df is not None:
        # --- Save the final dataset ---
        output_path = os.path.join(script_dir, "Data", "featured_data_final_v3.xlsx")
        print(f"\n💾 Saving data with all features to {output_path}...")
        
        try:
            featured_df.to_excel(output_path)
            print(f"✅ Final V3 featured data successfully saved!")
        except Exception as e:
            print(f"❌ Failed to save the file. Error: {e}")
