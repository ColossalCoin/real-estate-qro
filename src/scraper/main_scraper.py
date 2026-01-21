import asyncio
import re
import os
import random
import pandas as pd
from datetime import datetime
from playwright.async_api import async_playwright

# Update this path if your Chrome installation is different
CHROME_PATH = r"C:/chrome-win64/chrome.exe"
BASE_URL_ROOT = "https://www.vivanuncios.com.mx/s-casas-en-venta/queretaro/"
URL_SUFFIX = "v1c1293l1021p"
OUTPUT_DIR = "data/raw"
OUTPUT_FILENAME = "real_estate_queretaro_dataset_v2.csv"  # Changed name to avoid overwriting


class RealEstateScraper:
    """
    A professional-grade scraper for Real Estate data designed for Vivanuncios.

    Features:
    - Waterfall Strategy for Price Extraction (Metadata -> Title -> CSS -> Body).
    - Deep Attribute Parsing (Land vs. Construction Area, Parking spots).
    - Hidden Geolocation Mining.
    - Incremental Batch Saving (Fault tolerance).
    - **FULL DESCRIPTION CAPTURE** (Enabled for NLP Feature Engineering).
    """

    def __init__(self, headless=True, max_pages=1):
        self.headless = headless
        self.max_pages = max_pages
        self.browser = None
        self.context = None
        self.page = None
        self.playwright = None
        self.data_buffer = []

    @staticmethod
    def extract_number(text: str) -> float:
        """Utility to extract the first valid float number from a string."""
        if not text:
            return None
        match = re.search(r"(\d[\d,\.]*)", text)
        if match:
            clean_str = match.group(1).replace(",", "")
            try:
                return float(clean_str)
            except ValueError:
                return None
        return None

    async def start_browser(self):
        """Initializes the Playwright engine and browser context."""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [SYSTEM] Initializing Browser Engine...")

        executable_path = CHROME_PATH if os.path.exists(CHROME_PATH) else None

        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            executable_path=executable_path,
            headless=self.headless,
            args=["--disable-blink-features=AutomationControlled", "--start-maximized"]
        )
        self.context = await self.browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        self.page = await self.context.new_page()

    async def close_browser(self):
        """Performs a clean shutdown."""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [SYSTEM] Browser session closed.")

    async def parse_listing(self, url: str) -> dict:
        """
        Core logic to extract details from a single listing page.
        """
        try:
            await self.page.goto(url, timeout=45000)
            await self.page.wait_for_load_state("domcontentloaded")

            # 1. Raw Content Extraction & CLEANING
            body_handle = await self.page.locator("body").element_handle()
            full_text = await body_handle.inner_text()

            # --- CRITICAL UPDATE: Clean and Store Description ---
            # Collapse multiple spaces/newlines into single space
            cleaned_description = re.sub(r'\s+', ' ', full_text).strip()
            text_lower = full_text.lower()

            title = await self.page.title()
            html_content = await self.page.content()

            # 2. Price Extraction
            final_price = None

            def is_valid_price(val):
                return val is not None and val > 100000

            # LEVEL 0: Meta Tags (Highest Accuracy)
            if not final_price:
                meta_patterns = [
                    r'property="product:price:amount"\s+content="(\d+[\d\.]*)"',
                    r'name="price"\s+content="(\d+[\d\.]*)"',
                    r'"price":\s*(\d+[\d\.]*)'
                ]
                for pattern in meta_patterns:
                    match = re.search(pattern, html_content)
                    if match:
                        val = self.extract_number(match.group(1))
                        if is_valid_price(val):
                            final_price = val
                            break

            # LEVEL 1: Title Search
            if not final_price:
                title_matches = re.findall(r"\$\s?([\d,]+)", title)
                for match in title_matches:
                    val = self.extract_number(match)
                    if is_valid_price(val):
                        final_price = val
                        break

            # LEVEL 2: CSS Selectors
            if not final_price:
                selectors = ['span[class*="ad-price"]', 'div[class*="price"]', '[itemprop="price"]']
                for sel in selectors:
                    if await self.page.locator(sel).count() > 0:
                        raw_text = await self.page.locator(sel).first.inner_text()
                        val = self.extract_number(raw_text)
                        if is_valid_price(val):
                            final_price = val
                            break

            # LEVEL 3: Body Fallback
            if not final_price:
                body_matches = re.findall(r"(?:\$|MN)\s?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)", full_text)
                for match in body_matches:
                    val = self.extract_number(match)
                    if is_valid_price(val):
                        final_price = val
                        break

            # 3. Structural Attributes Extraction
            bedrooms, bathrooms, parking = None, None, None
            m2_constructed, m2_terrain = None, None

            # Summary Strip Regex
            match_lote = re.search(r"(\d+)\s?m²\s?lote", text_lower)
            if match_lote: m2_terrain = self.extract_number(match_lote.group(1))

            match_constr = re.search(r"(\d+)\s?m²\s?constr", text_lower)
            if match_constr: m2_constructed = self.extract_number(match_constr.group(1))

            match_park = re.search(r"(\d+)\s?estac", text_lower)
            if match_park: parking = self.extract_number(match_park.group(1))

            match_rec = re.search(r"(\d+)\s?rec", text_lower)
            if match_rec: bedrooms = self.extract_number(match_rec.group(1))

            match_bath = re.search(r"(\d+)\s?baños", text_lower)
            if match_bath: bathrooms = self.extract_number(match_bath.group(1))

            # Fallbacks if strip failed
            if not bedrooms:
                rec_match = re.search(r"(\d+)\s?(rec|hab|bed)", text_lower)
                bedrooms = float(rec_match.group(1)) if rec_match else None

            if not bathrooms:
                bath_match = re.search(r"(\d+\.?\d*)\s?(baño|bath)", text_lower)
                bathrooms = float(bath_match.group(1)) if bath_match else None

            # 4. Geolocation (Deep Extraction)
            lat, lon = None, None
            geo_patterns = [
                r"q=(-?\d+\.\d+),(-?\d+\.\d+)",
                r"&ll=(-?\d+\.\d+),(-?\d+\.\d+)",
                r'"latitude":\s*(-?\d+\.\d+).*?"longitude":\s*(-?\d+\.\d+)',
                r"center=(-?\d+\.\d+)%2C(-?\d+\.\d+)"
            ]
            for pattern in geo_patterns:
                match = re.search(pattern, html_content)
                if match:
                    try:
                        lat_c, lon_c = float(match.group(1)), float(match.group(2))
                        # Queretaro Bounding Box Filter
                        if 19.0 < lat_c < 22.0 and -101.5 < lon_c < -99.0:
                            lat, lon = lat_c, lon_c
                            break
                    except:
                        continue

            # 5. Location Text
            location_text = None
            try:
                loc_el = self.page.locator('span[class*="location"], div[class*="location"], [itemprop="address"]')
                if await loc_el.count() > 0:
                    location_text = await loc_el.first.inner_text()
            except:
                pass
            if not location_text and title:
                location_text = title

            return {
                "title": title,
                "price_numeric": final_price,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "parking_spots": parking,
                "m2_constructed": m2_constructed,
                "m2_terrain": m2_terrain,
                "latitude": lat,
                "longitude": lon,
                "location_text": location_text,
                "description": cleaned_description,  # <--- KEY ADDITION
                "url": url,
                "extraction_date": datetime.now().isoformat()
            }

        except Exception as e:
            print(f"[WARNING] Extraction failed for URL {url}: {str(e)}")
            return None

    def save_batch(self, new_data: list):
        """Saves a batch of data to the CSV file immediately."""
        if not new_data: return

        df = pd.DataFrame(new_data)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        csv_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
        file_exists = os.path.isfile(csv_path)

        try:
            df.to_csv(csv_path, index=False, mode='a', header=not file_exists, encoding='utf-8-sig')
            print(f"   [SYSTEM] 💾 Batch of {len(new_data)} records saved.")
        except Exception as e:
            print(f"   [ERROR] Save failed: {e}")

    async def run(self):
        """Main execution flow."""
        await self.start_browser()

        try:
            for current_page in range(1, self.max_pages + 1):
                target_url = f"{BASE_URL_ROOT}{URL_SUFFIX}{current_page}" if current_page == 1 else \
                    f"{BASE_URL_ROOT}page-{current_page}/{URL_SUFFIX}{current_page}"

                print(f"\n--> [Page {current_page}/{self.max_pages}] Navegando: {target_url}")
                batch_data = []

                try:
                    await self.page.goto(target_url, timeout=60000)

                    # Detect listings
                    card_selector = 'div[class*="postingCard-module__posting-container"]'
                    try:
                        await self.page.wait_for_selector(card_selector, timeout=20000)
                    except:
                        print(f"   [INFO] Pagination ended or blocked at page {current_page}.")
                        break

                    cards = await self.page.locator(card_selector).all()

                    # Extract URLs
                    page_urls = []
                    for card in cards:
                        link_locator = card.locator("a")
                        if await link_locator.count() > 0:
                            raw = await link_locator.first.get_attribute("href")
                            if raw:
                                full_link = f"https://www.vivanuncios.com.mx{raw}" if raw.startswith("/") else raw
                                if full_link not in page_urls:
                                    page_urls.append(full_link)

                    print(f"    Found {len(page_urls)} listings. Mining...")

                    # Process URLs
                    for i, link in enumerate(page_urls):
                        if i % 5 == 0: print(f"       Processing {i + 1}/{len(page_urls)}...")

                        data = await self.parse_listing(link)
                        if data:
                            data['source_page'] = current_page
                            batch_data.append(data)
                            self.data_buffer.append(data)

                        # Human-like delay
                        await self.page.wait_for_timeout(random.uniform(500, 1200))

                    self.save_batch(batch_data)

                except Exception as e:
                    print(f"[ERROR] Failed processing Page {current_page}: {e}")
                    continue

        finally:
            await self.close_browser()
            print(f"\n[FINISH] Process completed. Saved to {os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)}")


if __name__ == "__main__":
    print("--- Starting Real Estate Scraper Engine (Production v2) ---")
    scraper = RealEstateScraper(headless=True, max_pages=400)
    asyncio.run(scraper.run())