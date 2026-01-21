"""
Description:
    Feature Engineering Module.

    Responsibilities:
    1. Structural Linking: Generates a 'clean_address' Foreign Key for BigQuery Joins.
    2. NLP Extraction: Parses unstructured text descriptions to create binary
       amenity features (0/1) using Regular Expressions.

    Note: Input data is expected to have 'price_numeric' already pre-processed.
"""

import pandas as pd
import logging
import sys
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

# Import the AddressCleaner helper
try:
    from src.utils.clean_text import AddressCleaner
except ImportError:
    sys.path.append(str(BASE_DIR / "src" / "utils"))
    from clean_text import AddressCleaner

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Central class for enriching real estate data.
    Focuses on creating Foreign Keys and extracting NLP features.
    """

    def __init__(self, text_col='description', location_col='location_text'):
        # Configuration for column names
        self.text_col = text_col
        self.location_col = location_col

        # Define Regex Patterns for Amenities (Case Insensitive)
        self.amenity_patterns = {
            'has_security': [
                r'vigilancia', r'seguridad', r'cctv', r'control de acceso',
                r'port[oó]n el[eé]ctrico', r'caseta', r'guardia',
                r'circuito cerrado', r'privada'
            ],
            'has_garden': [
                r'jard[ií]n', r'patio trasero', r'amplio patio',
                r'[aá]reas? verdeg?s?', r'huerto', r'paisajismo'
            ],
            'has_pool': [
                r'alberca', r'piscina', r'carril de nado',
                r'jacuzzi', r'chapoteadero'
            ],
            'has_terrace': [
                r'terraza', r'roof garden', r'balc[oó]n',
                r'asador', r'palapa', r'solarium'
            ],
            'has_gym': [
                r'gimnasio', r'gym', r'ejercitadores'
            ],
            'is_new_property': [
                r'preventa', r'entrega inmediata', r'estrenar',
                r'acabados de lujo'
            ],
            'has_kitchen': [
                r'cocina integral', r'cocina equipada', r'granito'
            ]
        }

    def _normalize_text(self, series: pd.Series) -> pd.Series:
        """Helper: Normalizes text for NLP tasks."""
        return series.fillna("").astype(str).str.lower()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main orchestration method. Applies transformations to the dataframe.
        """
        logger.info(f"Starting Feature Extraction on {len(df)} records...")
        df_out = df.copy()

        # --- FOREIGN KEY GENERATION (Critical for BigQuery) ---
        if self.location_col in df_out.columns:
            logger.info("Generating 'clean_address' Foreign Key...")
            df_out['clean_address'] = df_out[self.location_col].apply(AddressCleaner.clean)
        else:
            logger.warning(f"Column '{self.location_col}' not found. Cannot generate Foreign Key.")

        # --- NLP AMENITY EXTRACTION ---
        if self.text_col in df_out.columns:
            logger.info("Extracting amenities from text descriptions...")
            search_space = self._normalize_text(df_out[self.text_col])

            for feature, keywords in self.amenity_patterns.items():
                regex_pattern = '|'.join(keywords)
                df_out[feature] = search_space.str.contains(
                    regex_pattern, case=False, regex=True
                ).astype(int)

            # --- Business Logic Corrections (Post-Processing) ---
            # Correcting "Service Patio" confusing the "Garden" logic
            mask_service_patio = search_space.str.contains(r'patio de (servicio|lavado|tendido)', regex=True)
            mask_strong_garden = search_space.str.contains(r'jard[ií]n|areas? verdeg?s?', regex=True)

            # Logic: If Garden=1 BUT it's a Service Patio AND NOT a real Garden -> Set Garden=0
            correction_mask = (df_out['has_garden'] == 1) & (mask_service_patio) & (~mask_strong_garden)

            if correction_mask.sum() > 0:
                df_out.loc[correction_mask, 'has_garden'] = 0
                logger.info(f"  > Refined Logic: Removed 'Service Patio' false positives from {correction_mask.sum()} rows.")
        else:
            logger.warning(f"Column '{self.text_col}' not found. Skipping NLP extraction.")

        return df_out


# --- EXECUTION ENTRY POINT ---
if __name__ == "__main__":
    INPUT_FILE = BASE_DIR / "data" / "raw" / "real_estate_queretaro_dataset.csv"
    OUTPUT_FILE = BASE_DIR / "data" / "processed" / "real_estate_enriched.csv"

    if not INPUT_FILE.exists():
        logger.error(f"Input file not found: {INPUT_FILE}")
    else:
        logger.info(f"Loading raw data from: {INPUT_FILE}")
        df_raw = pd.read_csv(INPUT_FILE)

        # Instantiate Extractor
        extractor = FeatureExtractor(
            text_col='description',
            location_col='location_text'
        )

        try:
            # Run Pipeline
            df_processed = extractor.transform(df_raw)

            # Save Output
            os.makedirs(OUTPUT_FILE.parent, exist_ok=True)
            df_processed.to_csv(OUTPUT_FILE, index=False)

            logger.info("--- SUCCESS ---")
            logger.info(f"Enriched dataset saved to: {OUTPUT_FILE}")

        except Exception as e:
            logger.critical(f"Feature Extraction Pipeline Failed: {e}")
            raise e