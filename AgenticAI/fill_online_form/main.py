
import json
import asyncio
from playwright.async_api import async_playwright
from openai import OpenAI
import time


client = OpenAI()

async def get_forms(page):
    """Extract all forms and their input fields on the current page."""
    forms = await page.query_selector_all("form")
    form_data = []
    for idx, form in enumerate(forms, 1):
        inputs = await form.query_selector_all("input, select, textarea")
        fields = []
        for inp in inputs:
            # Skip honeypot / invisible fields
            is_visible = await inp.is_visible()
            if not is_visible:
                continue
            field_info = {
                "name":           await inp.get_attribute("name") or "",
                "type":           await inp.get_attribute("type") or "text",
                "placeholder":    await inp.get_attribute("placeholder") or "",
                "required":       await inp.get_attribute("required") is not None,
                "label":          "",
                "data-format":    await inp.get_attribute("data-format") or "",
                "data-seperator": await inp.get_attribute("data-seperator") or "",
                "data-maxlength": await inp.get_attribute("data-maxlength") or "",
                "validate":       await inp.get_attribute("class") or "",
            }
            # Try to get label if exists
            field_id = await inp.get_attribute("id")
            if field_id:
                label = await page.query_selector(f'label[for="{field_id}"]')
                if label:
                    field_info["label"] = await label.inner_text()
            fields.append(field_info)
        form_data.append({"form_index": idx, "fields": fields})
    return form_data


def llm_decide_next_action(form_data, task_description="Extract all inputs"):
    """
    Use LLM to suggest next action:
    - which form to fill
    - what values to enter
    - whether to submit
    """
    prompt = f"""
You are an agent tasked to interact with a website form.

Current form data:
{form_data}

Task: {task_description}

Return a JSON with:
- action: "fill_form" / "submit_form" / "next_step" / "stop"
- form_index: which form to act on
- field_values: mapping from field names to values (only if action is fill_form)
"""
    
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        temperature=0,
    )
    try:
        return json.loads(response.output_text)
    except Exception:
        return {"action": "stop"}


def get_format_hint(field_type: str, field_meta: dict = None) -> str:
    """Return a human-readable format hint based on input type and field metadata."""
    # --- Detect JotForm lite date picker by placeholder or data-format attribute ---
    if field_meta:
        placeholder = field_meta.get("placeholder", "")
        data_format = field_meta.get("data-format", "")
        if placeholder in ("MM-DD-YYYY", "MM/DD/YYYY") or data_format == "mmddyyyy":
            return (
                "Format: MM-DD-YYYY            |  Example: 03-15-2024\n"
                "  Note   : Future dates only   |  Past dates are not allowed"
            )
    hints = {
        "text":           "Format: Any text string        |  Example: John Doe",
        "email":          "Format: user@example.com",
        "password":       "Format: Min 8 chars, mix of letters/numbers recommended",
        "number":         "Format: Numeric value          |  Example: 42",
        "tel":            "Format: Phone number           |  Example: +1-800-555-0199",
        "url":            "Format: Full URL               |  Example: https://example.com",
        "date":           "Format: YYYY-MM-DD             |  Example: 2024-03-15",
        "time":           "Format: HH:MM                  |  Example: 14:30",
        "datetime-local": "Format: YYYY-MM-DDTHH:MM       |  Example: 2024-03-15T14:30",
        "month":          "Format: YYYY-MM                |  Example: 2024-03",
        "week":           "Format: YYYY-WNN               |  Example: 2024-W10",
        "color":          "Format: Hex color code         |  Example: #ff5733",
        "range":          "Format: Numeric value within slider range",
        "checkbox":       "Format: y/yes/true/1 to check, anything else to uncheck",
        "radio":          "Format: Press Enter to select this option",
        "select":         "Format: Type the exact label of the option to select",
        "textarea":       "Format: Multi-line text (press Enter twice when done in terminal)",
        "file":           "Format: Absolute file path     |  Example: /home/user/file.pdf",
        "search":         "Format: Search query string",
    }
    return hints.get(field_type.lower(), "Format: Text input")


async def agentic_form_interaction(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # set True for headless
        page = await browser.new_page()
        await page.goto(url, wait_until="networkidle")
        #
        forms = await get_forms(page)
        if not forms:
            print("No forms found on this page.")
            await browser.close()
            return
        for form in forms:
            print(f"\n{'='*100}")
            print(f"  FORM {form['form_index']}")
            print(f"{'='*100}")
            fields = form["fields"]
            if not fields:
                print("No input fields found in this form.")
                continue
            for field in fields:
                if field["type"] == "hidden":
                    continue  # Skip hidden fields
                # Build a human-readable prompt for the field
                label       = field["label"] or field["name"] or field["placeholder"] or field["type"]
                field_type  = field["type"].lower()
                placeholder = field.get("placeholder", "")
                data_format = field.get("data-format", "")
                required_marker = "required" if field["required"] else "optional"
                fmt_hint    = get_format_hint(field_type, field_meta=field)
                # Locate the field on the page by name or placeholder
                selector = None
                if field["name"]:
                    selector = f'[name="{field["name"]}"]'
                elif field["placeholder"]:
                    selector = f'[placeholder="{field["placeholder"]}"]'
                if not selector:
                    print("*"*30)
                    print(f"Could not locate field '{label}' on page, skipping.")
                    print(str(field))
                    continue
                # Skip non-visible elements (honeypots, hidden inputs)
                if selector:
                    el_check = await page.query_selector(selector)
                    if el_check and not await el_check.is_visible():
                        continue
                # Pretty-print field metadata
                print("-"*50)
                print(f"Field      : {label}")
                print(f"required   : {required_marker}")
                print(f"Type       : {field_type}")
                print(f"placeholder: {placeholder}")
                print(f"data_format: {data_format}")
                print(f"{fmt_hint}")
                prompt = f"Enter > "
                user_input = input(prompt)
                element = await page.query_selector(selector)
                if not element:
                    print(f"  Element '{label}' not found on page, skipping.")
                    continue
                field_type = field["type"].lower()
                if field_type == "checkbox":
                    if user_input.strip().lower() in ("y", "yes", "true", "1"):
                        await element.check()
                    else:
                        await element.uncheck()
                elif field_type == "radio":
                    await element.check()
                elif field_type == "select":
                    await element.select_option(label=user_input)
                elif field_type == "file":
                    if user_input.strip():
                        await element.set_input_files(user_input.strip())
                    else:
                        print(f"  Skipping file upload for '{label}' (no path provided).")
                else:
                    await element.fill(user_input)
            # After filling all fields, ask whether to submit
            submit = input(f"\nSubmit form {form['form_index']}? (y/n): ").strip().lower()
            if submit == "y":
                submit_btn = await page.query_selector(
                    'form button[type="submit"], form input[type="submit"]'
                )
                if submit_btn:
                    await submit_btn.click()
                    await page.wait_for_load_state("networkidle")
                    print("Form submitted. Current URL:", page.url)
                else:
                    print("No submit button found. Attempting Enter key on last field...")
                    if element:
                        await element.press("Enter")
                        await page.wait_for_load_state("networkidle")
                        print("Done. Current URL:", page.url)
        await browser.close()


if __name__ == "__main__":
    test_url = "https://form.jotform.com/260497189942169"
    asyncio.run(agentic_form_interaction(test_url))


"""
python AgenticAI/fill_online_form/test_playwright2.py

"""
