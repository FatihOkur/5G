import geopandas as gpd
from shapely.geometry import Point
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt



def label_land_type_by_buffer(df, lat_col='Latitude', lon_col='Longitude', buffer_radius=5):
    """
    Her noktanın çevresindeki buffer alanı içinde hangi coğrafi yapı en yoğunsa onu 'land_type' olarak etiketler.
    """
    terrain_dir = "ITUMapData"
    metric_crs = "EPSG:3857"
    geo_crs = "EPSG:4326"

    # Geo veri setlerini yükle
    building = gpd.read_file(f"{terrain_dir}/ITU_3DBINA_EPSG4326.shp").to_crs(metric_crs)
    vegetation = gpd.read_file(f"{terrain_dir}/ITU_3DVEGETATION_EPSG4326.shp").to_crs(metric_crs)
    road = gpd.read_file(f"{terrain_dir}/ITU_ULASIMAGI_EPSG4326.shp").to_crs(metric_crs)
    water = gpd.read_file(f"{terrain_dir}/ITU_SUKUTLESI_EPSG4326.shp").to_crs(metric_crs)
    wall = gpd.read_file(f"{terrain_dir}/ITU_SINIRDUVAR_EPSG4326.shp").to_crs(metric_crs)

    # Noktaları GeoDataFrame'e çevir
    gdf = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs=geo_crs)
    gdf = gdf.to_crs(metric_crs)

    # 5 metrelik buffer alan oluştur
    gdf['buffer'] = gdf.geometry.buffer(buffer_radius)

    # Coğrafi yapıların yoğunluklarını hesapla
    land_types = ['building', 'vegetation', 'road', 'water', 'wall']
    sources = {
        'building': building,
        'vegetation': vegetation,
        'road': road,
        'water': water,
        'wall': wall
    }

    def get_dominant_type(buffer_geom):
        densities = {}
        for lt, src in sources.items():
            clipped = src.clip(buffer_geom)
            if clipped.empty:
                densities[lt] = 0
            else:
                # Yol ve duvar çizgi olduğu için uzunluk, diğerleri alan
                if clipped.geometry.iloc[0].geom_type in ['LineString', 'MultiLineString']:
                    densities[lt] = clipped.length.sum()
                else:
                    densities[lt] = clipped.area.sum()
        return max(densities, key=densities.get) if any(v > 0 for v in densities.values()) else 'other'

    gdf['land_type'] = gdf['buffer'].apply(get_dominant_type)
    gdf.drop(columns=['buffer'], inplace=True)

    return gdf


def plot_land_type_distributions(gdf_valid, train_idx, test_idx, output_path=None):
    """
    Genel, eğitim ve test kümelerindeki land_type dağılımlarını tek bir bar grafikte görselleştirir.
    """
    # Land type serilerini al
    all_series = gdf_valid['land_type']
    train_series = gdf_valid.loc[train_idx, 'land_type']
    test_series = gdf_valid.loc[test_idx, 'land_type']

    # Frekans tablolarını DataFrame'e çevir
    df_counts = pd.DataFrame({
        'Total': all_series.value_counts(),
        'Train': train_series.value_counts(),
        'Test': test_series.value_counts()
    }).fillna(0).astype(int).sort_index()

    # Plot
    ax = df_counts.plot(kind='bar', figsize=(10, 6), width=0.75)
    ax.set_title("Coğrafi Yapı Türlerinin Dağılımı (Genel vs. Train vs. Test)")
    ax.set_xlabel("Coğrafi Yapı Türü (land_type)")
    ax.set_ylabel("Örnek Sayısı")
    ax.legend(title="Küme")
    plt.xticks(rotation=0)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"✅ Dağılım grafiği kaydedildi: {output_path}")
    else:
        plt.show()



from sklearn.model_selection import train_test_split

def geo_balanced_train_test_split_with_buffer(df_filtered, lat_col='Latitude', lon_col='Longitude',
                                              test_size=0.2, random_state=42, buffer_radius=5,
                                              scaler=None, label_encoder=None, plot_path=None):
    """
    Coğrafi yapıya dayalı stratified test seti üretir. Veriler gridlenmiş ve filtrelenmiş olmalıdır.
    Eğitim seti kullanılmaz, sadece test seti döndürülür.
    
    Returns:
        X_test_scaled, y_test_encoded, test_indices (df_filtered içindeki index)
    """
    
    if scaler is None or label_encoder is None:
        raise ValueError("scaler and label_encoder must be provided.")

    # Coğrafi yapı türü etiketle
    gdf_labeled = label_land_type_by_buffer(df_filtered, lat_col=lat_col, lon_col=lon_col, buffer_radius=buffer_radius)

    # Geçerli verileri al
    valid_mask = gdf_labeled['land_type'].notnull()
    gdf_valid = gdf_labeled[valid_mask].copy()

    print("\n📊 Genel veri setindeki coğrafi yapı türü dağılımı:")
    print(gdf_valid['land_type'].value_counts())

    # Stratify için label
    stratify_labels = gdf_valid['land_type']

    # Test bölmesi (eğitim kullanılmayacak)
    _, X_test_raw, _, y_test_raw, _, test_idx = train_test_split(
        gdf_valid, gdf_valid['cell_id'], gdf_valid.index,
        test_size=test_size,
        stratify=stratify_labels,
        random_state=random_state
    )

    land_types_test = gdf_valid.loc[test_idx, 'land_type']
    land_types_train = gdf_valid.loc[gdf_valid.index.difference(test_idx), 'land_type']
    
    print("\n🧪 Test kümesindeki coğrafi yapı türü dağılımı:")
    print(land_types_test.value_counts())
    print("\n🎓 Eğitim kümesindeki coğrafi yapı türü dağılımı:")
    print(land_types_train.value_counts())

    # Görselleştir
    plot_land_type_distributions(gdf_valid, gdf_valid.index.difference(test_idx), test_idx, output_path=plot_path)

    # X ve y hazırla (eğitim modeli daha önceden eğitilmiş olduğu için sadece test verisi gerek)
    drop_cols = ['Latitude', 'Longitude', 'Time', 'cell_id', 'lat_cell', 'lon_cell', 'geometry', 'land_type']
    drop_cols_existing = [col for col in drop_cols if col in X_test_raw.columns]
    X_test = X_test_raw.drop(columns=drop_cols_existing, errors='ignore')
    X_test_scaled = scaler.transform(X_test)
    y_test_encoded = label_encoder.transform(y_test_raw)

    return X_test_scaled, y_test_encoded, test_idx


