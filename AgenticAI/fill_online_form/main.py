
import json
import asyncio
from playwright.async_api import async_playwright
from openai import OpenAI
import time


client = OpenAI()
model_name = "gpt-4.1-mini"


prompt_llm = """
You are given some user information and your task is to fill the values of each element in the given json object.
- Check the `label` carefully and make your answer based on it
- For each element, make a new item `value` and fill it with the right answer
- For each `value`, make sure to follow the `placeholder` or `data_format` format.
- If you dont have any information about the element, just fill it with blank.
- For the phone number, convert it to the `placeholder` formast.
- Make sure the retun json object is valid.

json elements:
{json_elements}

user info:
{user_info}
"""


user_info = """
name: James
family name: Abdi
phone number: +1222 333 4444
email: james.a@gmail.com
Position of interest: data science jobs

"""

json_schema = {
    "format": {
      "type": "json_schema",
      "name": "form_element_list",
      "strict": True,
      "schema": {
        "type": "object",
        "properties": {
          "items": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "form_index": {
                  "type": "integer",
                  "description": "The index of the form, starting from 1."
                },
                "elements": {
                  "type": "array",
                  "description": "A list of form elements (fields) in this form.",
                  "items": {
                    "type": "object",
                    "properties": {
                      "label": {
                        "type": "string",
                        "description": "Label shown for the form element."
                      },
                      "field_type": {
                        "type": "string",
                        "description": "Type of field (e.g. text, email, tel, file).",
                        "enum": [
                          "text",
                          "email",
                          "tel",
                          "file"
                        ]
                      },
                      "placeholder": {
                        "type": "string",
                        "description": "Placeholder text for the field."
                      },
                      "data_format": {
                        "type": "string",
                        "description": "Special data format constraints (if any); empty string if not set."
                      },
                      "is_visible": {
                        "type": "boolean",
                        "description": "Whether the form element is currently visible."
                      },
                      "value": {
                        "type": "string",
                        "description": "Current value of the form element (empty string if not set)."
                      }
                    },
                    "required": [
                      "label",
                      "field_type",
                      "placeholder",
                      "data_format",
                      "is_visible",
                      "value"
                    ],
                    "additionalProperties": False
                  }
                }
              },
              "required": [
                "form_index",
                "elements"
              ],
              "additionalProperties": False
            }
          }
        },
        "required": [
          "items"
        ],
        "additionalProperties": False
      }
    }
  }




def call_llm(prompt, json_schema):
    response = client.responses.create(
        model=model_name,
        input=prompt,
        temperature=0,
        reasoning={},
        tools=[],
        text=json_schema,
    )
    try:
        return json.loads(response.output_text)
    except Exception:
        return {}


async def get_forms(page, skip_hidden=True):
    """Extract all forms and their input fields on the current page."""
    forms = await page.query_selector_all("form")
    form_data = []
    for idx, form in enumerate(forms, 1):
        inputs = await form.query_selector_all("input, select, textarea")
        fields = []
        for inp in inputs:
            # Skip honeypot / invisible fields
            if skip_hidden:
                is_visible = await inp.is_visible()
                if not is_visible:
                    continue
            field_info = {
                "label":          "", # we will fill it if id exists
                "type":           await inp.get_attribute("type") or "text",
                "placeholder":    await inp.get_attribute("placeholder") or "",
                "required":       await inp.get_attribute("required") is not None,
                "field_id":       await inp.get_attribute("id"),
                "name":           await inp.get_attribute("name") or "",
                "data-format":    await inp.get_attribute("data-format") or "",
                "data-seperator": await inp.get_attribute("data-seperator") or "",
                "data-maxlength": await inp.get_attribute("data-maxlength") or "",
                "validate":       await inp.get_attribute("class") or "",
                "is_visible":     await inp.is_visible(),
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


async def form_get_elements(url, headless=True):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)  # set True for headless
        page = await browser.new_page()
        await page.goto(url, wait_until="networkidle")
        #
        forms = await get_forms(page)
        if not forms:
            print("No forms found on this page.")
            await browser.close()
            return
        #
        return forms


async def agentic_form_interaction(url, headless=False):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)  # set True for headless
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
    url = "https://form.jotform.com/260497189942169"
    all_elements = asyncio.run(form_get_elements(url))
    print("-"*80, "all_elements:")
    print(json.dumps(all_elements, indent=4))
    print("-"*80, "ret_llm:")
    prompt = prompt_llm.format(json_elements=all_elements, user_info=user_info)
    ret_llm = call_llm(prompt, json_schema)
    print(json.dumps(ret_llm, indent=4))





"""
python AgenticAI/fill_online_form/main.py





url = "https://form.jotform.com/260497189942169"
all_elements = asyncio.run(form_get_elements(url))
print("-"*80, "all_elements:")
print(json.dumps(all_elements, indent=4))

print("-"*80, "ret_llm:")
prompt = prompt_llm.format(json_elements=all_elements, user_info=user_info)
ret_llm = call_llm(prompt, json_schema)
print(json.dumps(ret_llm, indent=4))


prompt = prompt_llm.format(json_elements=all_elements, user_info=user_info)
response = client.responses.create(
    model=model_name,
    input=prompt,
    temperature=0,
    reasoning={},
    tools=[],
    text=json_schema,
)





"""
