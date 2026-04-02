# Panera Bread Order Bot

## Cookie File
Located at `~/panerabread_cookies.json`
- `sameSite` values from browser exports (e.g. `null`, `no_restriction`) must be normalized to Playwright-valid values (`Strict`, `Lax`, `None`)

## Task
Use Python with Playwright to:
1. Load cookies from the cookie file
2. Navigate to the beverages menu: https://www.panerabread.com/content/panerabread_com/en-us/menu/categories/beverages.html
3. Scrape and display the list of beverages
4. Ask the user what they want to order
5. Add selections to the order list and print a summary

## Environment
- Conda env: `panera_bread` (Python 3.11)
- Install deps: `pip install -r requirements.txt && playwright install chromium`

## Files
- `panera_order.py` — main script
- `requirements.txt` — dependencies (`playwright`)
- `README.md` — setup and usage instructions

## Known Issues Fixed
- `sameSite: null` in cookie JSON causes `AttributeError` — handled with `or "None"` fallback
- Browser `sameSite` format differs from Playwright's expected `Strict|Lax|None` — normalized via `SAME_SITE_MAP`
- Panera login page URL (e.g. `/en-us/mypanera/login.html`) returns 404 — navigate to homepage (`https://www.panerabread.com`) and let user sign in manually
