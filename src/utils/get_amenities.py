"""
Statewide POI Extractor (Modular & Configurable)
------------------------------------------------
Extracts Points of Interest (POIs) for Querétaro state.
Separates extraction logic from storage logic.
"""

import logging
import pandas as pd
import osmnx as ox
from pathlib import Path

STATE_NAME = "Querétaro, Mexico"

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "external"
OUTPUT_FILENAME = "amenities.csv"

# Geographic Scope
MUNICIPALITIES = [
    "Amealco de Bonfil", "Arroyo Seco", "Cadereyta de Montes", "Colón",
    "Corregidora", "Ezequiel Montes", "Huimilpan", "Jalpan de Serra",
    "Landa de Matamoros", "El Marqués", "Pedro Escobedo", "Peñamiller",
    "Pinal de Amoles", "Querétaro", "San Joaquín", "San Juan del Río",
    "Tequisquiapan", "Tolimán"
]

# Feature Selection
STATEWIDE_TAGS = {
    'amenity': ['hospital', 'clinic', 'university', 'school', 'kindergarten', 'pharmacy', 'townhall', 'marketplace'],
    'leisure': ['park', 'playground', 'garden', 'pitch', 'sports_centre'],
    'landuse': ['industrial', 'recreation_ground', 'grass'],
    'shop': ['supermarket', 'mall', 'department_store', 'convenience'],
    'tourism': ['attraction', 'hotel', 'museum']
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_municipal_centers() -> pd.DataFrame:
    """Finds coordinates for administrative centers."""
    centers = []
    logger.info("Locating Municipal Centers...")

    for muni in MUNICIPALITIES:
        query = f"Centro, {muni}, Querétaro, Mexico"
        try:
            lat, lon = ox.geocode(query)
            centers.append({
                "name": f"Centro {muni}",
                "category": "municipal_center",
                "latitude": lat,
                "longitude": lon
            })
        except Exception as e:
            logger.warning(f"Skipping {muni}: {e}")

    return pd.DataFrame(centers)

def categorize_poi(row: pd.Series) -> str:
    """Applies classification rules to raw data."""
    # Macro
    if row.get('landuse') == 'industrial': return 'hub_industrial'
    if row.get('tourism') in ['attraction', 'museum']: return 'hub_tourism'
    if row.get('shop') in ['mall', 'department_store']: return 'hub_commercial'

    # Micro - Health & Retail
    if row.get('amenity') == 'hospital': return 'health_hospital'
    if row.get('amenity') in ['clinic', 'pharmacy']: return 'health_local'
    if row.get('shop') == 'supermarket': return 'shop_supermarket'
    if row.get('shop') == 'convenience': return 'shop_convenience'
    if row.get('amenity') == 'marketplace': return 'shop_market'

    # Micro - Lifestyle & Education
    if row.get('leisure') in ['park', 'garden']: return 'nature_park'
    if row.get('leisure') == 'playground': return 'nature_playground'
    if row.get('landuse') in ['recreation_ground', 'grass']: return 'nature_green_area'
    if row.get('amenity') in ['university', 'college']: return 'education_university'
    if row.get('amenity') in ['school', 'kindergarten']: return 'education_school'

    # Gov
    if row.get('amenity') == 'townhall': return 'municipal_center'

    return 'other_service'


def extract_infrastructure() -> pd.DataFrame:
    """Main ETL function: Extracts, Cleans, and Transforms OSM data."""
    try:
        logger.info(f"Downloading tags from OSM for {STATE_NAME}...")
        gdf = ox.features_from_place(STATE_NAME, tags=STATEWIDE_TAGS)
    except Exception as e:
        logger.error(f"OSM Download failed: {e}")
        return pd.DataFrame()

    if gdf.empty: return pd.DataFrame()

    # 1. GEOMETRY TO POINTS
    gdf['centroid'] = gdf.geometry.centroid
    gdf['latitude'] = gdf['centroid'].y
    gdf['longitude'] = gdf['centroid'].x

    # 2. CATEGORIZATION
    for col in ['landuse', 'tourism', 'amenity', 'shop', 'leisure']:
        if col not in gdf.columns: gdf[col] = None

    gdf['category'] = gdf.apply(categorize_poi, axis=1)

    # 3. CLEANING NAMES (La parte que preguntaste)
    # Fill empty names with category-based generic names
    if 'name' not in gdf.columns: gdf['name'] = None

    gdf['name'] = gdf.apply(
        lambda x: str(x['name']).strip() if pd.notnull(x['name']) and str(x['name']).strip() != ''
        else f"Unnamed {x['category'].replace('_', ' ').title()}",
        axis=1
    )

    # 4. BRAND STANDARDIZATION (Bonus: Oxxo, Oxo -> Oxxo)
    gdf['name'] = gdf['name'].replace(['Oxo', 'oxo', 'OXXO'], 'Oxxo')

    # 5. DEDUPLICATION
    # Remove POIs that are in the exact same spot
    initial_count = len(gdf)
    clean_df = gdf[['name', 'category', 'latitude', 'longitude']].drop_duplicates(
        subset=['latitude', 'longitude']
    ).copy()

    logger.info(f"Removed {initial_count - len(clean_df)} duplicate POIs.")

    return clean_df

def save_dataset(df: pd.DataFrame, directory: Path, filename: str):
    """
    Handles file persistence. Isolate this to allow future changes
    (e.g., saving to S3/GCS) without touching logic code.
    """
    if df.empty:
        logger.warning("No data to save.")
        return

    directory.mkdir(parents=True, exist_ok=True)
    file_path = directory / filename

    try:
        df.to_csv(file_path, index=False)
        logger.info(f"Success! Data saved to: {file_path}")
        logger.info(f"Rows: {len(df)}")
    except Exception as e:
        logger.error(f"Failed to save file: {e}")


def main():
    logger.info("Pipeline Started")

    # 1. Extract Logic
    df_infra = extract_infrastructure()
    df_centers = get_municipal_centers()

    # 2. Merge Logic
    if not df_infra.empty:
        df_final = pd.concat([df_infra, df_centers], ignore_index=True)
        # Clean coordinates
        df_final = df_final.dropna(subset=['latitude', 'longitude'])
    else:
        df_final = df_centers



    # 3. Save Logic (Decoupled)
    save_dataset(df_final, DATA_DIR, OUTPUT_FILENAME)

if __name__ == "__main__":
    main()