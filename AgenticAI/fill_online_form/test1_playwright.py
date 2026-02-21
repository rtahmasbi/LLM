from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    browser = p.chromium.launch()
    print("Browser works!")
    browser.close()


"""
python AgenticAI/fill_online_form/test_playwright1.py

"""