import asyncio
from playwright.async_api import async_playwright
import os
import sys

CHROME_PATH = r"C:\chrome-win64\chrome.exe"
# Using a specific search query to ensure results are populated
TARGET_URL = "https://www.vivanuncios.com.mx/s-casas-en-venta/queretaro/v1c1293l1021p1"


async def inspect_listing_structure():
    """
    Performs a structural analysis of the first listing card found in the DOM.
    Useful for debugging selector issues and understanding parent-child relationships
    without executing a full scrape cycle.
    """

    if not os.path.exists(CHROME_PATH):
        print(f"[ERROR] Chrome executable not found at: {CHROME_PATH}")
        sys.exit(1)

    async with async_playwright() as p:
        print("[INFO] Initializing Browser Engine for Inspection...")

        # Launch browser in visible mode for manual verification if needed
        browser = await p.chromium.launch(
            executable_path=CHROME_PATH,
            headless=False,
            args=["--disable-blink-features=AutomationControlled"]
        )

        page = await browser.new_page()

        try:
            print(f"[INFO] Navigating to target URL: {TARGET_URL}")
            await page.goto(TARGET_URL, timeout=60000)

            # Selector previously identified via reverse engineering
            card_selector = 'div[class*="postingCard-module__posting-container"]'

            print(f"[INFO] Waiting for selector: {card_selector}")
            await page.wait_for_selector(card_selector, timeout=20000)

            # Capture all instances
            cards = await page.locator(card_selector).all()
            count = len(cards)

            if count > 0:
                print("\n" + "=" * 60)
                print(f"DOM INSPECTION REPORT - {count} CARDS DETECTED")
                print("=" * 60)

                # Analyze the first card instance
                first_card = cards[0]

                # A. Content Preview
                text_content = await first_card.inner_text()
                preview = text_content[:100].replace('\n', ' ')
                print(f"[DATA] First Card Content Preview: '{preview}...'")

                # B. Internal Link Analysis (Anchors nested inside the div)
                internal_links = await first_card.locator("a").all()
                print(f"[STRUCTURE] Internal <a> tags found: {len(internal_links)}")

                for i, link in enumerate(internal_links):
                    href = await link.get_attribute("href")
                    print(f"    - Link index [{i}] href: '{href}'")

                # C. Parent Hierarchy Analysis
                # Check if the container is wrapped by an anchor tag (common in React apps)
                parent_locator = first_card.locator("..")
                parent_tag = await parent_locator.evaluate("el => el.tagName")

                print(f"[STRUCTURE] Immediate Parent Tag: <{parent_tag}>")

                if parent_tag == "A":
                    parent_href = await parent_locator.get_attribute("href")
                    print(f"[INSIGHT] The card is wrapped in a link. Parent href: '{parent_href}'")
                else:
                    print("[INSIGHT] The card is a standalone container (not wrapped in <a>).")

            else:
                print("[WARNING] Selector was valid but no elements were returned.")

        except Exception as e:
            print(f"[ERROR] Inspection failed during execution: {e}")

        finally:
            print("[INFO] Closing browser session.")
            await browser.close()


if __name__ == "__main__":
    asyncio.run(inspect_listing_structure())