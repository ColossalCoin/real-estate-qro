import asyncio
from playwright.async_api import async_playwright
import os
import sys

CHROME_PATH = r"C:\chrome-win64\chrome.exe"
# Specific URL to ensure elements are present for inspection
TARGET_URL = "https://www.vivanuncios.com.mx/s-casas-en-venta/queretaro/v1c1293l1021p1"


async def launch_interactive_inspector():
    """
    Launches the browser in a paused state with the Playwright Inspector GUI enabled.
    This allows for manual selection of DOM elements to retrieve their CSS selectors/XPaths.

    Usage:
    1. Run this script.
    2. Wait for the browser and the 'Playwright Inspector' window to open.
    3. Click the 'Pick Locator' (or 'Explore') button in the Inspector.
    4. Hover over any element on the website to see its selector.
    5. Copy the selector and update the main scraper if necessary.
    6. Press 'Resume' in the Inspector to close the session.
    """

    if not os.path.exists(CHROME_PATH):
        print(f"[ERROR] Chrome binary not found at: {CHROME_PATH}")
        sys.exit(1)

    async with async_playwright() as p:
        print("[INFO] Initializing Interactive Inspector Environment...")

        # Launch browser with GUI enabled and stealth args
        browser = await p.chromium.launch(
            executable_path=CHROME_PATH,
            headless=False,
            args=["--disable-blink-features=AutomationControlled", "--start-maximized"]
        )

        # Create context with standard viewport and user agent
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080}
        )

        page = await context.new_page()

        try:
            print(f"[INFO] Navigating to: {TARGET_URL}")
            await page.goto(TARGET_URL, timeout=60000)

            print("\n" + "=" * 60)
            print(" SYSTEM PAUSED FOR INSPECTION ")
            print("=" * 60)
            print("Instructions:")
            print("1. Look for the 'Playwright Inspector' window.")
            print("2. Click the 'Pick locator' button.")
            print("3. Hover over elements in the browser to inspect attributes.")
            print("4. Press the 'Resume' (Play icon) in the Inspector to exit.")
            print("=" * 60 + "\n")

            # This halts execution and hands control to the Playwright Inspector GUI
            await page.pause()

        except Exception as e:
            print(f"[ERROR] Session interrupted: {e}")

        finally:
            print("[INFO] Closing browser session.")
            await browser.close()


if __name__ == "__main__":
    asyncio.run(launch_interactive_inspector())