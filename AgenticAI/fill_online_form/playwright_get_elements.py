
import json
import asyncio
from playwright.async_api import async_playwright, Page, Browser
import argparse



# ── Helpers ───────────────────────────────────────────────────────────────────

async def _extract_fields_from_context(page: Page, context, skip_hidden: bool = True):
    """
    Extract fields from a form element or the full page.
    Handles:
      - Duplicate IDs (appends _dup1, _dup2, ...)
      - Signature divs (skipped — not real inputs)
      - Checkbox siblings (flagged for mutual-exclusion handling)
    """
    inputs = await context.query_selector_all("input, select, textarea")
    fields = []
    seen_ids: dict[str, int] = {}  # original_id -> count seen so far
    
    for inp in inputs:
        if skip_hidden and not await inp.is_visible():
            continue
        # field_id and effective_id
        field_id = await inp.get_attribute("id")
        # Deduplicate IDs
        if field_id:
            if field_id in seen_ids:
                seen_ids[field_id] += 1
                effective_id = f"{field_id}_dup{seen_ids[field_id]}"
            else:
                seen_ids[field_id] = 0
                effective_id = field_id
        else:
            effective_id = None
        # is_required
        is_required = await inp.evaluate("el => el.required")
        if not is_required:
            aria_required = await inp.get_attribute("aria-required") or ""
            if aria_required.lower() == "true":
                is_required = True
        if not is_required:
            class_attr = await inp.get_attribute("class") or ""
            if "required" in class_attr.lower():
                is_required = True        
        # field_type
        field_type = await inp.get_attribute("type") or "text"
        role = await inp.get_attribute("role") or ""
        aria_haspopup = await inp.get_attribute("aria-haspopup") or ""
        if role == "combobox" or aria_haspopup in ("true", "listbox"):
            field_type = "select"
        # field_name
        field_name  = await inp.get_attribute("name") or ""
        field_info = {
            "label":          "",
            "type":           field_type,
            "placeholder":    await inp.get_attribute("placeholder") or "",
            "id":             effective_id,
            "original_id":    field_id,
            "name":           field_name,
            "data-format":    await inp.get_attribute("data-format") or "",
            "data-seperator": await inp.get_attribute("data-seperator") or "",
            "data-maxlength": await inp.get_attribute("data-maxlength") or "",
            "is_visible":     await inp.is_visible(),
            "is_required":    is_required,
            # Flag checkbox groups that share the same name (radio-like behaviour)
            "exclusive_group": field_name if field_type == "checkbox" else "",
        }
        # ── Resolve label text — try multiple strategies ──────────────────
        label_text = ""
        # 1. <label for="id"> — standard HTML
        if field_id and not label_text:
            el = await page.query_selector(f'label[for="{field_id}"]')
            if el:
                label_text = (await el.inner_text()).strip()
        # 2. Immediate next sibling that is a <label> or <span>
        #    Covers: <input .../><label>text</label>
        #            <input .../><span>text</span>
        if not label_text:
            label_text = await inp.evaluate("""el => {
                let sib = el.nextElementSibling;
                if (sib && (sib.tagName === 'LABEL' || sib.tagName === 'SPAN'))
                    return sib.innerText.trim();
                return '';
            }""")
        # 3. Parent container text — walk up to the nearest .checkbox-group /
        #    .form-row / li / div and grab all text excluding the input's own value.
        #    This catches patterns like:
        #      <div class="checkbox-group">
        #        <input type="checkbox" id="x"/>
        #        <span>Long descriptive label text…</span>
        #      </div>
        if not label_text:
            label_text = await inp.evaluate("""el => {
                const stopTags = new Set(['FORM','BODY','HTML','TABLE','TR','TD','TH']);
                let node = el.parentElement;
                while (node && !stopTags.has(node.tagName)) {
                    const txt = node.innerText.trim();
                    if (txt.length > 0) return txt;
                    node = node.parentElement;
                }
                return '';
            }""")
            # If the parent text is very long it probably contains the label +
            # other sibling fields; take only the first 300 chars to stay useful.
            if label_text and len(label_text) > 300:
                label_text = label_text[:300].rsplit(' ', 1)[0] + '…'
        # 4. aria-label / aria-labelledby fallback
        if not label_text:
            label_text = await inp.get_attribute("aria-label") or ""
        if not label_text:
            labelledby = await inp.get_attribute("aria-labelledby") or ""
            if labelledby:
                el = await page.query_selector(f'[id="{labelledby}"]')
                if el:
                    label_text = (await el.inner_text()).strip()
        #
        field_info["label"] = label_text
        fields.append(field_info)
    #
    return fields


async def get_forms(page: Page, skip_hidden: bool = True):
    """
    Extract all forms and their input fields on the current page.
    Falls back to scanning the whole page when no <form> tags are present
    (covers JS-rendered and iframe-less single-page apps).
    Also checks iframes if no forms/inputs are found on the main frame.
    """
    forms = await page.query_selector_all("form")
    # ── Iframe fallback ──────────────────────────────────────────────────────
    if not forms:
        for frame in page.frames:
            if frame == page.main_frame:
                continue
            try:
                iframe_forms = await frame.query_selector_all("form")
                if iframe_forms:
                    form_data = []
                    for idx, form in enumerate(iframe_forms, 1):
                        fields = await _extract_fields_from_context(frame, form, skip_hidden)
                        form_data.append({"form_index": idx, "fields": fields, "source": "iframe"})
                    return form_data
            except Exception:
                continue
    # ── No <form> tags at all: scan whole page for inputs ───────────────────
    if not forms:
        fields = await _extract_fields_from_context(page, page, skip_hidden)
        if fields:
            return [{"form_index": 1, "fields": fields, "source": "page_scan"}]
        return [{"error": "No form or input elements found on page after waiting."}]
    # ── Normal path ──────────────────────────────────────────────────────────
    form_data = []
    for idx, form in enumerate(forms, 1):
        fields = await _extract_fields_from_context(page, form, skip_hidden)
        form_data.append({"form_index": idx, "fields": fields, "source": "form"})
    return form_data



async def main(url, headless=False):
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
        print(json.dumps(forms, indent=2))
        await browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Form filler agent")
    parser.add_argument("--url",       required=True, help="Target form URL")
    parser.add_argument("--headless",  default=True,
                        type=lambda x: x.lower() != "false",
                        help="Headless browser? Pass --headless false to watch it run.")
    args      = parser.parse_args()
    asyncio.run(main(args.url, args.headless))





"""
cd AgenticAI/fill_online_form/
python playwright_get_elements.py \
  --url https://www.uber.com/global/en/careers/list/153458/ \
  --headless false


headless = True
test_url = "https://form.jotform.com/260497189942169"
asyncio.run(agentic_form_interaction(test_url, headless=headless))



###### sync version
from playwright.sync_api import sync_playwright

url = "https://www.uber.com/global/en/careers/list/153458/"
headless = False

p = sync_playwright().start()
browser = p.chromium.launch(headless=False)
page = browser.new_page()
page.goto("https://www.uber.com/global/en/careers/list/153458/")


page.get_by_role("link", name="Apply Now").first.click()
page.get_by_role("button", name="Sign in").click()

page.get_by_test_id("email-input-id").fill("your_email@example.com")



# Click the initial Sign in button
page.get_by_test_id("signin-button").click()

# Wait for email input
#page.get_by_test_id("email-input-id").wait_for()

# Fill credentials
page.get_by_test_id("email-input-id").fill("")
page.get_by_test_id("password-input-id").fill("")

# Click submit button
page.locator('button[type="submit"]').click()

# Wait for navigation or state change
page.wait_for_load_state("networkidle")


error_locators = [
        page.locator('[role="alert"]'),
        page.locator('[data-testid*="error" i]'),
        page.locator('text=/invalid|incorrect|wrong|try again|error/i'),
        page.locator('input[name="email"][aria-invalid="true"]'),
        page.locator('input[name="password"][aria-invalid="true"]'),
]

timeout_ms=15000
page.wait_for_function(
            /"/""() => {
                const hasAlert = document.querySelector('[role="alert"]');
                const ariaInvalid = document.querySelector('input[aria-invalid="true"]');
                const email = document.querySelector('[data-testid="email-input-id"]');
                // if email input is gone, likely success
                return !!hasAlert || !!ariaInvalid || !email;
            }"/"/",
            timeout=timeout_ms,
)




##################### sync version
from playwright.sync_api import sync_playwright

url = "https://job-boards.greenhouse.io/doordashusa/jobs/6338077#app"

p = sync_playwright().start()
browser = p.chromium.launch(headless=False)
page = browser.new_page()
page.goto(url)


page.get_by_role("link", name="Apply To Job").first.click()
page.get_by_role("link", name="Apply").first.click()



python playwright_get_elements.py \
  --url https://job-boards.greenhouse.io/doordashusa/jobs/6338077#app \
  --headless false


<div class="select__container">
   <label id="question_51375307-label" for="question_51375307" class="label select__label">How much did content from the DoorDash Engineering blog influence your decision to apply for a role at DoorDash?
   <span aria-hidden="true">*</span>
   </label>
   <div class="select-shell remix-css-b62m3t-container">
      <span id="react-select-question_51375307-live-region" class="remix-css-7pg0cj-a11yText"></span>
      <span aria-live="polite" aria-atomic="false" aria-relevant="additions text" role="log" class="remix-css-7pg0cj-a11yText"></span>
      <div>
         <div class="select__control remix-css-13cymwt-control">
            <div class="select__value-container remix-css-hlgwow">
               <div class="select__placeholder remix-css-1jqq78o-placeholder" id="react-select-question_51375307-placeholder">Select...</div>
               <div class="select__input-container remix-css-19bb58m" data-value="">
                  <input class="select__input" style="label:input;color:inherit;background:0;opacity:1;width:100%;grid-area:1 / 2;font:inherit;min-width:2px;border:0;margin:0;outline:0;padding:0" autocapitalize="none" autocomplete="off" autocorrect="off" id="question_51375307" spellcheck="false" tabindex="0" type="text" aria-autocomplete="list" aria-expanded="false" aria-haspopup="true" aria-errormessage="question_51375307-error" aria-invalid="false" aria-labelledby="question_51375307-label" aria-required="true" role="combobox" aria-activedescendant="" aria-describedby="react-select-question_51375307-placeholder question_51375307-error" enterkeyhint="done" value="">
               </div>
            </div>
            <div class="select__indicators remix-css-1wy0on6">
               <button type="button" class="icon-button icon-button--sm" aria-label="Toggle flyout" tabindex="-1">
                  <svg class="svg-icon" fill="none" height="20" width="20" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                     <path class="icon--primary-color" d="M11.4534 16.0667L5.90983 9.13729C5.54316 8.67895 5.86948 8 6.45644 8H17.5436C18.1305 8 18.4568 8.67895 18.0902 9.13729L12.5466 16.0667C12.2664 16.417 11.7336 16.417 11.4534 16.0667Z"></path>
                  </svg>
               </button>
            </div>
         </div>
      </div>
      <input required="" tabindex="-1" aria-hidden="true" class="remix-css-1a0ro4n-requiredInput" value="">
   </div>
</div>

"""
