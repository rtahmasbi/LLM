# Panera Bread Beverage Ordering Script

Uses Python with Playwright to browse the Panera Bread beverages menu and interactively build an order.

## Setup

```bash
# Create and activate conda env
conda create -n panera_bread python=3.11 -y
conda activate panera_bread

# Install dependencies
pip install -r requirements.txt
playwright install chromium
```

## Run

```bash
python panera_order.py
```

## What it does

1. Loads cookies from `/Users/rasooltahmasbi/panerabread_cookies.json` (if present)
2. Navigates to the Panera beverages menu page using Playwright
3. Scrapes and displays the menu items in a numbered list
4. Prompts you to enter item numbers to add to your order
5. Prints a final order summary

## Files

- `panera_order.py` — main script
- `requirements.txt` — Python dependencies
- `README.md` — this file
