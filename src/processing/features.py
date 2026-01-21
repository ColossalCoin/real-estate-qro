"""
Description:
    Parses unstructured text descriptions from real estate listings to extract
    structured binary features (Amenities) using Regular Expressions (NLP).

    This module is part of the ETL pipeline, bridging the gap between raw web-scraped
    data and the Machine Learning model.
"""

import pandas as pd
import logging
import os
from pathlib import Path

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts binary amenities (0/1) from text descriptions.
    """

    def __init__(self):
        # Define Regex Patterns (Case Insensitive)
        # Note: We use Spanish keywords because the raw data is in Spanish.
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
        """
        Normalizes text: lowercase, fills NA, removes special chars if needed.
        """
        return series.fillna("").astype(str).str.lower()

    def transform(self, df: pd.DataFrame, text_col: str = 'description') -> pd.DataFrame:
        """
        Main method to apply transformations.

        Args:
            df (pd.DataFrame): Raw dataframe containing the text column.
            text_col (str): The name of the column to parse.

        Returns:
            pd.DataFrame: DataFrame with new binary columns attached.
        """
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found in DataFrame.")

        logger.info(f"Starting feature extraction on {len(df)} records...")

        # Working on a copy to avoid SettingWithCopy warnings
        df_out = df.copy()

        # Pre-process text once for efficiency
        search_space = self._normalize_text(df_out[text_col])

        # Iterating through the dictionary of patterns
        for feature, keywords in self.amenity_patterns.items():
            # Create a single compiled regex pattern: (word1|word2|word3)
            # This is significantly faster than looping through keywords.
            regex_pattern = '|'.join(keywords)

            # Vectorized search
            df_out[feature] = search_space.str.contains(
                regex_pattern, case=False, regex=True
            ).astype(int)

            count = df_out[feature].sum()
            logger.info(f"  > Feature '{feature}': Detected in {count} listings ({count / len(df):.1%})")

        # --- BUSINESS LOGIC CORRECTIONS ---
        # Fix 1: "Patio de servicio" (Laundry area) is NOT a Garden.
        # We check if 'has_garden' is 1, but the text explicitly mentions service patio.

        # Define mask for service patio
        mask_service_patio = search_space.str.contains(r'patio de (servicio|lavado|tendido)', regex=True)

        # If it has garden AND matches service patio pattern, we need to be careful.
        # We only remove it if it DOESN'T match strong garden keywords (like 'jardin' or 'areas verdes').
        mask_strong_garden = search_space.str.contains(r'jard[ií]n|areas? verdeg?s?', regex=True)

        # Logic: If it has "Patio" but also "Patio de servicio" and NO "Jardin", set to 0.
        correction_mask = (df_out['has_garden'] == 1) & (mask_service_patio) & (~mask_strong_garden)

        n_corrected = correction_mask.sum()
        if n_corrected > 0:
            df_out.loc[correction_mask, 'has_garden'] = 0
            logger.info(f"  > Applied Logic Fix: Removed 'Patio de servicio' false positives from {n_corrected} rows.")

        logger.info("Feature extraction completed successfully.")
        return df_out


# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    # Define paths
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    INPUT_FILE = BASE_DIR / "data" / "raw" / "real_estate_queretaro_dataset.csv"
    OUTPUT_FILE = BASE_DIR / "data" / "processed" / "real_estate_enriched.csv"

    # Check if file exists
    if not INPUT_FILE.exists():
        logger.error(f"Input file not found: {INPUT_FILE}")
    else:
        logger.info(f"Loading data from: {INPUT_FILE}")
        df_raw = pd.read_csv(INPUT_FILE)

        # Initialize and Transform
        extractor = FeatureExtractor()
        try:
            df_processed = extractor.transform(df_raw)

            # Save output
            os.makedirs(OUTPUT_FILE.parent, exist_ok=True)
            df_processed.to_csv(OUTPUT_FILE, index=False)
            logger.info(f"Data saved to: {OUTPUT_FILE}")

        except Exception as e:
            logger.critical(f"Pipeline failed: {e}")