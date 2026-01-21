import pandas as pd
import time
import logging
import sys
import os
from pathlib import Path
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

try:
    from src.utils.clean_text import AddressCleaner
except ImportError:
    from clean_text import AddressCleaner

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class GeocoderService:
    """
    Handles interactions with the OpenStreetMap API (Nominatim).
    Includes logic for:
    - Rate limiting (to avoid bans)
    - Fallback strategies (handling private clusters like 'Zikura, Zibata')
    """

    def __init__(self, app_name="qro_real_estate_analyzer_v1"):
        # Setup Client: Nominatim requires a unique user_agent
        self.geolocator = Nominatim(user_agent=app_name, timeout=10)

        # Setup Rate Limiter: Crucial to prevent 400 Errors (Too Many Requests)
        # Nominatim asks for 1 second delay minimum.
        self.geocode_api = RateLimiter(self.geolocator.geocode, min_delay_seconds=1.1)

        # 3. Context
        self.macro_context = ", Querétaro, México"

    def _query_api(self, query_string):
        """
        Internal helper to execute the safe API call.
        """
        try:
            # Add context to avoid finding "Jurica" in another country
            full_query = f"{query_string}{self.macro_context}"
            return self.geocode_api(full_query)
        except (GeocoderTimedOut, GeocoderUnavailable):
            logger.warning(f"API Timeout on: {query_string}. Retrying...")
            time.sleep(2)
            return None
        except Exception as e:
            logger.error(f"API Error on '{query_string}': {e}")
            return None

    def get_coordinates(self, clean_address):
        """
        Main logic: Tries to find coordinates implementing a Fallback Strategy.
        Assumes 'clean_address' is ALREADY cleaned.
        Returns: (latitude, longitude)
        """
        if not clean_address or pd.isna(clean_address):
            return None, None

        # Step 1: Primary Attempt (Exact Match)
        location = self._query_api(clean_address)

        # Step 2: Fallback Strategy
        # If exact match fails AND there is a comma, try searching for the parent neighborhood.
        if not location and ',' in clean_address:
            # Logic: Split "zikura, zibatá" -> take "zibatá"
            parts = clean_address.split(',')
            if len(parts) > 1:
                broader_address = parts[-1].strip()
                if len(broader_address) > 3:
                    # logger.info(f"   ↳ Retry broader: '{broader_address}'")
                    location = self._query_api(broader_address)

        if location:
            return location.latitude, location.longitude
        return None, None

    def process_batch(self, df: pd.DataFrame, address_col: str) -> pd.DataFrame:
        """
        Applies geocoding to a DataFrame with progress tracking.
        """
        logger.info(f"Starting Geocoding for {len(df)} unique locations...")
        logger.info("   This process includes rate limiting (1.1s delay). Please wait.")

        df_out = df.copy()

        # Show real-time progress on the console
        lats, lons = [], []
        total = len(df_out)

        for i, idx, row in enumerate(df_out.iterrows()):
            addr = row[address_col]
            lat, lon = self.get_coordinates(addr)
            lats.append(lat)
            lons.append(lon)

            # Progress bar
            if i % 5 == 0 or i == total - 1:
                sys.stdout.write(f"\rProcessing: {i + 1}/{total} - Last: {addr[:20]}...")
                sys.stdout.flush()

        print("\n")
        df_out['latitude'] = lats
        df_out['longitude'] = lons

        success_rate = df_out['latitude'].notnull().mean()
        logger.info(f"Geocoding Complete. Success Rate: {success_rate:.1%}")

        return df_out


if __name__ == "__main__":
    INPUT_FILE = BASE_DIR / "data" / "raw" / "real_estate_queretaro_dataset.csv"
    OUTPUT_FILE = BASE_DIR / "data" / "processed" / "geo_catalog.csv"

    if not INPUT_FILE.exists():
        logger.error(f"Input file not found: {INPUT_FILE}")
        logger.info("Please run the scraper (main_scraper.py) first to generate data.")
    else:
        logger.info(f"Loading raw data from: {INPUT_FILE}")
        df_raw = pd.read_csv(INPUT_FILE)
        raw_col_name = 'location_text'

        if raw_col_name in df_raw.columns:
            # Apply AddressCleaner logic
            df_raw['clean_address'] = df_raw[raw_col_name].apply(AddressCleaner.clean)

            # Create a unique list of addresses to minimize API calls
            df_catalog = df_raw[['clean_address']].dropna().drop_duplicates()

            logger.info(f"Optimization: Reduced {len(df_raw)} raw rows to {len(df_catalog)} unique locations.")

            # GEOCODING EXECUTION
            geocoder = GeocoderService()
            # Pass the newly created 'clean_address' column
            df_result = geocoder.process_batch(df_catalog, address_col='clean_address')

            # EXPORT RESULTS
            os.makedirs(OUTPUT_FILE.parent, exist_ok=True)
            df_result.to_csv(OUTPUT_FILE, index=False)
            logger.info(f"Geocoded catalog saved successfully to: {OUTPUT_FILE}")

        else:
            logger.critical(f"Column '{raw_col_name}' not found. Available columns: {list(df_raw.columns)}")