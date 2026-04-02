import argparse
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright
from playwright_stealth import Stealth

PROFILE_DIR = Path.home() / ".panera_profile"
HOME_URL = "https://www.panerabread.com"
BEVERAGES_URL = "https://www.panerabread.com/content/panerabread_com/en-us/menu/categories/beverages.html"

LAUNCH_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--no-sandbox",
    "--disable-dev-shm-usage",
]

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


async def new_stealth_page(context):
    page = await context.new_page()
    await Stealth().apply_stealth_async(page)
    return page


async def open_persistent_context(playwright, headless):
    return await playwright.chromium.launch_persistent_context(
        str(PROFILE_DIR),
        channel="chrome",
        headless=headless,
        args=LAUNCH_ARGS,
        user_agent=USER_AGENT,
    )


async def _login_flow():
    async with async_playwright() as p:
        print("Opening browser for sign-in...")
        context = await open_persistent_context(p, headless=False)
        page = await new_stealth_page(context)
        await page.goto(HOME_URL, wait_until="networkidle", timeout=250000)
        input()  # safe — no running event loop
        await context.close()
        print("Sign-in complete. Profile saved.")


def ensure_logged_in():
    """If no profile exists yet, open browser for manual login."""
    if PROFILE_DIR.exists():
        print(f"Profile found at {PROFILE_DIR} — skipping login.")
        return

    print("\nNo browser profile found. A Chrome window will open.")
    print("Step 1: Enter your email and click Continue.")
    print("Step 2: Enter your password and click Sign In.")
    print("Once fully signed in, press Enter here...")
    asyncio.run(_login_flow())


async def get_beverages(playwright, headless=True):
    context = await open_persistent_context(playwright, headless=headless)
    page = await new_stealth_page(context)

    print("\nNavigating to beverages menu...")
    await page.goto(BEVERAGES_URL, wait_until="networkidle", timeout=150000)

    items = []
    selectors = [
        "[data-testid='menu-item-name']",
        ".menu-item__title",
        ".menu-product-name",
        "h3.product-name",
        "[class*='menuItem'] [class*='name']",
        "[class*='MenuItem'] [class*='name']",
        "h2, h3, h4",
    ]

    for selector in selectors:
        elements = await page.query_selector_all(selector)
        if elements:
            for el in elements:
                text = (await el.inner_text()).strip()
                if text and len(text) > 2:
                    items.append(text)
            if items:
                break

    seen = set()
    unique_items = []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique_items.append(item)

    await context.close()
    return unique_items


def display_menu(items):
    print("\n=== Panera Bread Beverages Menu ===")
    if not items:
        print("No items found. The page structure may have changed.")
        return
    for i, item in enumerate(items, 1):
        print(f"  {i:>2}. {item}")
    print("===================================\n")


def take_order(items):
    order = []
    print("Enter the number(s) of items you want to order (comma-separated), or type 'done' to finish:")

    while True:
        user_input = input("> ").strip()
        if user_input.lower() in ("done", "q", "quit", "exit"):
            break

        selections = [s.strip() for s in user_input.split(",")]
        for sel in selections:
            if sel.isdigit():
                idx = int(sel) - 1
                if 0 <= idx < len(items):
                    item = items[idx]
                    order.append(item)
                    print(f"  Added: {item}")
                else:
                    print(f"  Invalid number: {sel}")
            else:
                print(f"  Please enter a valid number (got '{sel}')")

        print("\nCurrent order:", order if order else "(empty)")
        print("Add more items or type 'done' to finish:")

    return order


def show_order_summary(order):
    print("\n=== Your Order ===")
    if not order:
        print("No items ordered.")
    else:
        for i, item in enumerate(order, 1):
            print(f"  {i}. {item}")
    print("==================")


async def main(headless=True):
    print(f"Starting Panera Bread order bot (headless={headless})...")
    async with async_playwright() as p:
        items = await get_beverages(p, headless=headless)

    display_menu(items)

    if items:
        order = take_order(items)
        show_order_summary(order)
    else:
        print("Could not retrieve menu items. Please check the page or your network connection.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Panera Bread beverage ordering script")
    parser.add_argument(
        "--headless",
        type=lambda x: x.lower() != "false",
        default=True,
        metavar="true|false",
        help="Run browser in headless mode (default: true)",
    )
    args = parser.parse_args()
    ensure_logged_in()
    asyncio.run(main(headless=args.headless))
