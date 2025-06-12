import geopandas as gpd
from shapely.geometry import Point, LineString
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from shapely.geometry import Polygon
import numpy as np
import pandas as pd
from classification_model import create_grid_and_assign_cells

def save_predictions_to_shapefile(pred_coords, true_coords, output_path):
    """
    Given predicted and true coordinates, save them as point and line geometries in a shapefile.

    Parameters:
        pred_coords (np.ndarray): Predicted coordinates (lat, lon) as numpy array.
        true_coords (np.ndarray): True coordinates (lat, lon) as numpy array.
        output_path (str): Directory where shapefiles will be saved.
    """
    os.makedirs(output_path, exist_ok=True)

    # Create GeoDataFrame for predicted points
    pred_points = [Point(lon, lat) for lat, lon in pred_coords]
    gdf_pred = gpd.GeoDataFrame(geometry=pred_points, crs="EPSG:4326")
    gdf_pred["type"] = "predicted"
    gdf_pred.to_file(os.path.join(output_path, "predicted_points.shp"))

    # Create GeoDataFrame for true points
    true_points = [Point(lon, lat) for lat, lon in true_coords]
    gdf_true = gpd.GeoDataFrame(geometry=true_points, crs="EPSG:4326")
    gdf_true["type"] = "true"
    gdf_true.to_file(os.path.join(output_path, "true_points.shp"))

    # Optional: Save line paths for visual comparison
    if len(pred_coords) == len(true_coords):
        pred_line = LineString([Point(lon, lat) for lat, lon in pred_coords])
        true_line = LineString([Point(lon, lat) for lat, lon in true_coords])
        gdf_lines = gpd.GeoDataFrame({
            "type": ["predicted_path", "true_path"],
            "geometry": [pred_line, true_line]
        }, crs="EPSG:4326")
        gdf_lines.to_file(os.path.join(output_path, "paths.shp"))

    print(f"✅ Shapefiles saved to: {output_path}")


def draw_grid_on_map(cell_centers, grid_size_meters=10, output_path="grid_map.png", background_shp_path=None):
    """
    Draws the generated grid cells onto a map and saves it as an image.

    Args:
        cell_centers (dict): Dictionary mapping cell_id to (lat, lon) coordinates of the cell center.
        grid_size_meters (int): The size of each grid cell in meters.
        output_path (str): Path to save the output image file.
        background_shp_path (str, optional): Path to a shapefile to use as a map background.
    """
    print(f"🗺️  Visualizing grid with {len(cell_centers)} cells...")

    # Re-calculate grid properties
    lat_min, lat_max = 41.098692000, 41.110922000
    lon_min, lon_max = 29.014443000, 29.037912000
    lat_step = grid_size_meters / 111000
    mean_latitude_campus = (lat_min + lat_max) / 2
    lon_step = grid_size_meters / (111000 * np.cos(np.radians(mean_latitude_campus)))

    fig, ax = plt.subplots(1, 1, figsize=(15, 15))

    if background_shp_path:
        try:
            campus_map = gpd.read_file(background_shp_path)
            campus_map.plot(ax=ax, color='lightgray', edgecolor='black')
            print(f"✅ Background map '{background_shp_path}' loaded.")
        except Exception as e:
            print(f"⚠️  Could not load background shapefile: {e}. Skipping background.")

    for cell_id, (center_lat, center_lon) in cell_centers.items():
        bottom_left_lon = center_lon - (lon_step / 2)
        bottom_left_lat = center_lat - (lat_step / 2)
        rect = Rectangle(
            (bottom_left_lon, bottom_left_lat),
            lon_step,
            lat_step,
            linewidth=0.5,
            edgecolor='red',
            facecolor='red',
            alpha=0.2
        )
        ax.add_patch(rect)

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f'ITU Campus Grid ({grid_size_meters}m x {grid_size_meters}m cells)', fontsize=16)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Grid map successfully saved to '{output_path}'")

def save_grid_to_shapefile(cell_centers, grid_size_meters=10, output_path="grid.shp"):
    """
    Saves the generated grid cells as a polygon shapefile.

    Args:
        cell_centers (dict): Dictionary mapping cell_id to (lat, lon) coordinates.
        grid_size_meters (int): The size of each grid cell in meters.
        output_path (str): Path to save the output shapefile.
    """
    print(f"💾 Saving grid with {len(cell_centers)} cells to shapefile...")

    # Re-calculate grid properties to define polygon boundaries
    lat_min, lat_max = 41.098692000, 41.110922000
    lon_min, lon_max = 29.014443000, 29.037912000
    lat_step = grid_size_meters / 111000
    mean_latitude_campus = (lat_min + lat_max) / 2
    lon_step = grid_size_meters / (111000 * np.cos(np.radians(mean_latitude_campus)))
    
    geometries = []
    cell_ids = []

    for cell_id, (center_lat, center_lon) in cell_centers.items():
        # Calculate the four corners of the polygon for this cell
        bottom_left = (center_lon - lon_step / 2, center_lat - lat_step / 2)
        bottom_right = (center_lon + lon_step / 2, center_lat - lat_step / 2)
        top_right = (center_lon + lon_step / 2, center_lat + lat_step / 2)
        top_left = (center_lon - lon_step / 2, center_lat + lat_step / 2)
        
        # Create a Shapely Polygon object
        geometries.append(Polygon([bottom_left, bottom_right, top_right, top_left]))
        cell_ids.append(cell_id)

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame({'cell_id': cell_ids, 'geometry': geometries}, crs="EPSG:4326")
    
    # Save to shapefile
    try:
        gdf.to_file(output_path)
        print(f"✅ Grid shapefile successfully saved to '{output_path}'")
    except Exception as e:
        print(f"❌ Failed to save shapefile: {e}")


if __name__ == '__main__':
    # Load your actual data
    df = pd.read_excel("5G/Data/Merged_encoded_filled_filtered_pciCleaned_featureEngineered.xlsx")

    # 1. Create the grid and get the cell centers dictionary
    df_gridded, cell_centers = create_grid_and_assign_cells(df, grid_size_meters=10)

    """# 2. Call the function to draw the map visualization
    boundary_shapefile = "5G/ITUMapData/ITU_SINIRDUVAR_EPSG4326.shp"
    draw_grid_on_map(
        cell_centers, 
        grid_size_meters=50, 
        output_path="campus_grid_visualization.png", 
        background_shp_path=boundary_shapefile
    )"""
    
    # 3. Call the new function to save the grid as a shapefile
    save_grid_to_shapefile(
        cell_centers,
        grid_size_meters=10,
        output_path="campus_grid.shp"
    )




