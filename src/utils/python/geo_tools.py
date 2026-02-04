"""
Description:
    Geospatial Data Utilities.

    This module handles format conversions for geospatial files, specifically
    preparing standard GeoJSON files for ingestion into Google BigQuery.

    BigQuery requires JSON data to be Newline Delimited (JSONL/NDJSON),
    where each line is a distinct JSON object, rather than a single
    nested JSON array.
"""

import json
import logging
from pathlib import Path
from typing import Union

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_geojson_to_jsonl(input_path: Union[str, Path], output_path: Union[str, Path]) -> None:
    """
    Reads a standard GeoJSON FeatureCollection and writes each Feature
    as a separate line in a JSONL file.

    Args:
        input_path (Path): Path to the source .geojson file.
        output_path (Path): Path where the .jsonl file will be saved.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    try:
        logger.info(f"Reading GeoJSON: {input_path.name}...")

        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Validate GeoJSON Structure
        if 'features' not in data:
            logger.error("Invalid GeoJSON: Root object must contain a 'features' list.")
            return

        features = data['features']
        total_features = len(features)
        logger.info(f"Found {total_features} features (polygons). Converting...")

        # Write to JSONL (Newline Delimited JSON)
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for i, feature in enumerate(features):
                # Ensure geometry is valid before writing
                if 'geometry' in feature and feature['geometry']:
                    json.dump(feature, f_out)
                    f_out.write('\n')  # The critical newline character
                else:
                    logger.warning(f"Skipping feature index {i}: Missing geometry.")

        logger.info(f"Conversion successful.")
        logger.info(f"Output saved to: {output_path}")

    except json.JSONDecodeError:
        logger.error("Failed to decode JSON. The source file might be corrupted.")
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    INPUT_FILE = BASE_DIR / "data" / "external" / "22_Queretaro.json"
    OUTPUT_FILE = BASE_DIR / "data" / "processed" / "municipios_bq.jsonl"

    # Create processed directory if it doesn't exist
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Run Conversion
    convert_geojson_to_jsonl(INPUT_FILE, OUTPUT_FILE)