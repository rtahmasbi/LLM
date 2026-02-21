
import json
import asyncio
from playwright.async_api import async_playwright
from openai import OpenAI
import time


client = OpenAI()
model_name = "gpt-4.1-mini"
max_output_tokens = 2048



prompt_llm = """
You are given some user information and your task is to fill the values of each element in the given json object.
- Check the `label` carefully and make your answer based on it
- For each element, fill the `value` with the right answer
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

json_schema =  text={
    "format": {
      "type": "json_schema",
      "name": "form_list",
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
                  "description": "Index of the form; 1-based position."
                },
                "fields": {
                  "type": "array",
                  "description": "List of fields for the form.",
                  "items": {
                    "type": "object",
                    "properties": {
                      "label": {
                        "type": "string",
                        "description": "Field label (can be empty)."
                      },
                      "type": {
                        "type": "string",
                        "description": "Type of the form field (text, email, tel, file, etc.)."
                      },
                      "placeholder": {
                        "type": "string",
                        "description": "Placeholder text for input (can be empty)."
                      },
                      "id": {
                        "anyOf": [
                          {
                            "type": "string"
                          },
                          {
                            "type": "null"
                          }
                        ],
                        "description": "Field identifier; can be null when missing."
                      },
                      "name": {
                        "type": "string",
                        "description": "Name attribute for the field (can be empty)."
                      },
                      "data-format": {
                        "type": "string",
                        "description": "Field format value (can be empty)."
                      },
                      "data-seperator": {
                        "type": "string",
                        "description": "Separator value (can be empty)."
                      },
                      "data-maxlength": {
                        "type": "string",
                        "description": "Maximum length for input (as string, can be empty)."
                      },
                      "validate": {
                        "type": "string",
                        "description": "Validate class or rule string."
                      },
                      "is_visible": {
                        "type": "boolean",
                        "description": "Is the field visible on the form."
                      },
                      "value": {
                        "type": "string",
                        "description": "The value for the Field."
                      }
                    },
                    "required": [
                      "label",
                      "type",
                      "placeholder",
                      "id",
                      "name",
                      "data-format",
                      "data-seperator",
                      "data-maxlength",
                      "validate",
                      "is_visible",
                      "value",
                    ],
                    "additionalProperties": False
                  }
                }
              },
              "required": [
                "form_index",
                "fields"
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
        max_output_tokens=max_output_tokens,
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
                "id":             await inp.get_attribute("id"),
                "name":           await inp.get_attribute("name") or "",
                "data-format":    await inp.get_attribute("data-format") or "",
                "data-seperator": await inp.get_attribute("data-seperator") or "",
                "data-maxlength": await inp.get_attribute("data-maxlength") or "",
                "validate":       await inp.get_attribute("class") or "",
                "is_visible":     await inp.is_visible(),
            }
            # Try to get label if exists
            id = await inp.get_attribute("id")
            if id:
                label = await page.query_selector(f'label[for="{id}"]')
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


async def agentic_form_interaction(ret_llm, url, headless=True):
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
        for idx_form, form in enumerate(forms):
            print(f"\n{'='*100}", f"FORM {form['form_index']}")
            fields = form["fields"]
            if not fields:
                print("No input fields found in this form.")
                continue
            for field in fields:
                if field["type"] == "hidden":
                    continue  # Skip hidden fields
                # Locate the field on the page by name or placeholder
                selector = None
                id = field["id"]
                if id:
                    selector = f'[id="{id}"]'
                if not selector:
                    print("*"*30)
                    print(f"Could not locate field id '{id}' on page, skipping.")
                    print(str(field))
                    continue
                # Skip non-visible elements (honeypots, hidden inputs)
                if selector:
                    el_check = await page.query_selector(selector)
                    if el_check and not await el_check.is_visible():
                        continue
                # Pretty-print field metadata
                print("-"*50)
                print(f"id   : {id}")
                print(f"label: {field['label']}")
                #prompt = f"Enter > "
                #user_input = input(prompt)
                user_input = get_filed_value(ret_llm, idx_form, id)
                print(f"value: {user_input}")
                element = await page.query_selector(selector)
                if not element:
                    print(f"  Element with id '{id}' not found on page, skipping.")
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
                        print(f"  Skipping file upload for id '{id}' (no path provided).")
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



def get_filed_value(ret_llm, idx_form, id):
    form_fields = ret_llm["items"][idx_form]["fields"]
    for f in form_fields:
        if f["id"] == id:
            return f["value"]
    return ""
    


if __name__ == "__main__":
    url = "https://form.jotform.com/260497189942169"
    all_elements = asyncio.run(form_get_elements(url))
    print("-"*80, "all_elements:")
    print(json.dumps(all_elements, indent=4))
    print("-"*80, "ret_llm:")
    prompt = prompt_llm.format(json_elements=all_elements, user_info=user_info)
    ret_llm = call_llm(prompt, json_schema)
    print(json.dumps(ret_llm, indent=4))
    print("-"*80, "agentic_form_interaction:")
    all_elements = asyncio.run(agentic_form_interaction(ret_llm, url))





"""
python AgenticAI/fill_online_form/test3_playwright_with_llm.py





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
