import geopandas as gpd
from shapely.geometry import Point, LineString
import os

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
