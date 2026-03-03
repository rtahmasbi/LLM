import json
import asyncio
from playwright.async_api import async_playwright, Page, Browser

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langgraph.graph import StateGraph, MessagesState, END, START
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel


from typing import Optional, List
import argparse



# ── Shared browser session ────────────────────────────────────────────────────

class BrowserSession:
    """Holds a single Playwright browser + page that lives for the whole run."""
    def __init__(self):
        self._playwright = None
        self._browser: Optional[Browser] = None
        self.page: Optional[Page] = None

    async def start(self, headless: bool = True):
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=headless)
        self.page = await self._browser.new_page()
        print("[BrowserSession] Browser started.")

    async def goto(self, url: str):
        """Navigate only if not already on that URL."""
        if self.page.url != url:
            # Use domcontentloaded (faster) then wait for JS frameworks to render
            await self.page.goto(url, wait_until="domcontentloaded")
            await self.page.wait_for_timeout(2500)
            print(f"[BrowserSession] Navigated to {url}")
        else:
            print(f"[BrowserSession] Already on {url}, skipping navigation.")

    async def close(self):
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        print("[BrowserSession] Browser closed.")


class Element(BaseModel):
    field_id: str = ""
    value: str = ""
    is_required: bool = True


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

        is_required = await inp.evaluate("el => el.required")
        if not is_required:
            class_attr = await inp.get_attribute("class") or ""
            if "required" in class_attr.lower():
                is_required = True

        field_type = await inp.get_attribute("type") or "text"
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

        # Resolve label via for= attribute (use original_id)
        if field_id:
            label_el = await page.query_selector(f'label[for="{field_id}"]')
            if label_el:
                field_info["label"] = await label_el.inner_text()

        fields.append(field_info)

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


async def _select_best_option(element, user_input: str) -> None:
    """
    Try to select a <select> option using multiple strategies in order:
      1. Exact label match   (e.g. "California")
      2. Exact value match   (e.g. "CA")
      3. Case-insensitive label match
      4. Partial label match (first option whose text contains the input)
    Raises ValueError if nothing matches.
    """
    raw = user_input.strip()

    # Collect all options as (value, label) pairs
    options: list[tuple[str, str]] = await element.evaluate("""
        el => Array.from(el.options).map(o => [o.value, o.text.trim()])
    """)

    # 1. Exact label
    for val, label in options:
        if label == raw:
            await element.select_option(value=val)
            return

    # 2. Exact value
    for val, label in options:
        if val == raw:
            await element.select_option(value=val)
            return

    # 3. Case-insensitive label
    raw_lower = raw.lower()
    for val, label in options:
        if label.lower() == raw_lower:
            await element.select_option(value=val)
            return

    # 4. Partial label (first match)
    for val, label in options:
        if raw_lower in label.lower():
            await element.select_option(value=val)
            return

    raise ValueError(
        f"No option matching '{raw}'. "
        f"Available: {[label for _, label in options]}"
    )


async def _fill_page_flat(page: Page, values: List[Element]) -> str:
    """
    Fill fields from a list of Element objects and return a summary.

    Handles:
      - Duplicate IDs via _dup<N> suffix (queries all matching elements and
        picks the right occurrence).
      - Checkbox-as-radio groups: unchecks all siblings sharing the same
        `name` attribute before checking the target.
      - Signature <div> placeholders: skipped gracefully.
    """
    summary_lines = []

    for v in values:
        id_ = v.field_id
        user_input = v.value
        is_required = v.is_required

        # ── Resolve original id and occurrence index ─────────────────────
        if "_dup" in id_:
            original_id, dup_str = id_.rsplit("_dup", 1)
            try:
                dup_index = int(dup_str)
            except ValueError:
                dup_index = 0
        else:
            original_id = id_
            dup_index = 0

        all_matching = await page.query_selector_all(f'[id="{original_id}"]')
        if not all_matching or dup_index >= len(all_matching):
            summary_lines.append(f"  [SKIP] id='{id_}' — not found (occurrence {dup_index})")
            continue

        element = all_matching[dup_index]

        if not await element.is_visible():
            summary_lines.append(f"  [SKIP] id='{id_}' — not visible")
            continue

        field_type = (await element.get_attribute("type") or "text").lower()
        tag = await element.evaluate("el => el.tagName.toLowerCase()")

        try:
            match field_type:
                case "checkbox":
                    want_checked = user_input.strip().lower() in ("y", "yes", "true", "1")

                    # Uncheck all siblings in the same name-group first
                    # (handles checkbox-used-as-radio pattern)
                    field_name = await element.get_attribute("name") or ""
                    if field_name:
                        siblings = await page.query_selector_all(
                            f'input[type="checkbox"][name="{field_name}"]'
                        )
                        for sibling in siblings:
                            await sibling.uncheck()

                    if want_checked:
                        await element.check()

                case "radio":
                    await element.check()

                case "file":
                    if user_input.strip():
                        await element.set_input_files(user_input.strip())
                    else:
                        summary_lines.append(f"  [SKIP] file upload for id='{id_}' — no path given")
                        continue

                case _:
                    if tag == "select":
                        await _select_best_option(element, user_input)
                    else:
                        await element.fill(user_input)

            summary_lines.append(
                f"  [OK]   id='{id_}' | value='{user_input}' | is_required={is_required}"
            )

        except Exception as e:
            summary_lines.append(f"  [ERR]  id='{id_}' — {e}")

    return "\n".join(summary_lines)


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
async def form_get_elements(url: str, config: RunnableConfig) -> list:
    """
    Extract all visible form fields from a webpage.
    Returns a list of forms, each containing field metadata
    (id, original_id, label, type, placeholder, is_required, exclusive_group).
    Always call this first before filling any form.

    Notes:
      - Duplicate IDs are renamed with a _dup<N> suffix so every field has a
        unique addressable id.
      - Fields with no <input>/<select>/<textarea> (e.g. signature divs) are
        not returned.
      - Checkboxes sharing the same `name` are flagged with exclusive_group;
        the agent should treat them as mutually exclusive (radio-like).
    """
    session: BrowserSession = config["configurable"]["browser_session"]
    await session.goto(url)

    # Wait for either a <form> or at least an <input> — whichever appears first.
    try:
        await session.page.wait_for_selector("form, input, select, textarea", timeout=15000)
    except Exception as e:
        return [{"error": f"No form elements detected after 15 s: {e}"}]

    return await get_forms(session.page)


@tool
def form_validate_elements(url: str, values: List[Element], config: RunnableConfig) -> str:
    """
    Checks that all required fields have a non-empty value.
    Returns "True" when valid, or a descriptive error string listing the
    missing required field IDs.
    Always call this before form_fill_fields.
    """
    missing = [v.field_id for v in values if v.is_required and not v.value.strip()]
    if missing:
        return f"False — missing required fields: {', '.join(missing)}"
    return "True"


@tool
async def form_fill_fields(url: str, values: List[Element], config: RunnableConfig) -> str:
    """
    Fill form fields on a webpage. Does NOT submit — returns a summary for
    human review and saves a screenshot to /tmp/form_preview.png.

    Args:
        url: The page URL.
        values: List of Element(field_id, value, is_required).
                Use the field ids returned by form_get_elements (including
                _dup<N> suffixes for duplicate ids).
    """
    session: BrowserSession = config["configurable"]["browser_session"]
    await session.goto(url)
    summary = await _fill_page_flat(session.page, values)
    screenshot_path = "/tmp/form_preview.png"
    await session.page.screenshot(path=screenshot_path, full_page=True)
    return summary + f"\n\n[Preview screenshot saved to {screenshot_path}]"


@tool
async def form_submit(url: str, values: List[Element], config: RunnableConfig) -> str:
    """
    Re-fill form fields and SUBMIT the form.
    Call this ONLY after the human has approved submission.

    Args:
        url: The page URL.
        values: Same list of Element objects used in form_fill_fields.
    """
    session: BrowserSession = config["configurable"]["browser_session"]
    current = session.page.url
    if not current.startswith(url.rstrip("/")):
        await session.goto(url)
        await _fill_page_flat(session.page, values)

    submit_btn = await session.page.query_selector(
        'form button[type="submit"], form input[type="submit"]'
    )
    if submit_btn:
        await submit_btn.click()
    else:
        await session.page.keyboard.press("Enter")

    await session.page.wait_for_load_state("networkidle")
    screenshot_path = "/tmp/final_state.png"
    await session.page.screenshot(path=screenshot_path, full_page=True)
    print(f"Final page screenshot saved to {screenshot_path}")
    return f"Form submitted successfully. Final URL: {session.page.url}"


# ── Human approval middleware ─────────────────────────────────────────────────

async def human_approval_node(state: MessagesState) -> Command:
    fill_summary = next(
        (m.content for m in reversed(state["messages"])
         if isinstance(m, ToolMessage) and "form_fill_fields" in (m.name or "")),
        "No fill summary found.",
    )

    human_decision = interrupt({
        "question": (
            "The form has been filled. Please review the field values below "
            "and decide whether to submit.\n\n"
            f"{fill_summary}\n\n"
            "Reply 'yes' to submit or 'no' to cancel."
        )
    })

    fill_args = next(
        (tc["args"]
         for m in reversed(state["messages"]) if hasattr(m, "tool_calls")
         for tc in m.tool_calls if tc["name"] == "form_fill_fields"),
        {},
    )

    if str(human_decision).strip().lower() in ("yes", "y"):
        return Command(
            goto="agent",
            update={"messages": state["messages"] + [HumanMessage(
                content=(
                    "Human approved.\nNow call `form_submit` with "
                    f"url=\"{fill_args.get('url', '')}\" and "
                    f"values={json.dumps(fill_args.get('values', {}))}."
                )
            )]},
        )
    else:
        return Command(
            goto=END,
            update={"messages": state["messages"] + [
                HumanMessage(content="User declined. Task cancelled.")
            ]},
        )


# ── LLM + graph ───────────────────────────────────────────────────────────────

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


SYSTEM_PROMPT = """
You are a form-filling assistant. Your workflow is strictly:

1. Call `form_get_elements` to extract all visible fields from the URL.
2. Using the extracted field ids and the user info provided, build a list with
   values=[{"field_id":field_id, "is_required":is_required, "value":value}] and call `form_validate_elements`.
   - Include ALL fields returned, even optional ones, setting value="" if no data was provided.
   - For fields with duplicate IDs (suffixed _dup1, _dup2 ...), treat each as a separate field.
   - For checkbox groups sharing the same exclusive_group name, only set value="yes" for the
     intended option and leave the rest as value="no".
   - Skip any field whose id is None (non-input elements).
3. If `form_validate_elements` returns "True", call `form_fill_fields` with the SAME values and url.
   If `form_validate_elements` returns "False", STOP immediately and inform the user which
   required fields are missing. Do NOT call form_fill_fields.
4. WAIT — a human approval step will pause execution here.
5. If approved, call `form_submit` with the SAME url and values.
6. Report the final outcome.

Rules:
- Never skip step 1.
- Never call `form_submit` before human approval.
- Never call `form_fill_fields` if validation failed.
- Signature fields (type=div / no id) are not fillable — do not include them.
"""


async def call_agent(state: MessagesState):
    all_tools = [form_get_elements, form_validate_elements, form_fill_fields, form_submit]
    response = await llm.bind_tools(all_tools).ainvoke(
        [{"role": "system", "content": SYSTEM_PROMPT}] + state["messages"]
    )
    return {"messages": [response]}

def should_continue(state: MessagesState):
    last = state["messages"][-1]
    if not hasattr(last, "tool_calls") or not last.tool_calls:
        return END
    match last.tool_calls[0]["name"]:
        case "form_get_elements":       return "tools_extract"
        case "form_validate_elements":  return "tools_validate"
        case "form_fill_fields":        return "tools_fill"
        case _:                         return "tools_submit"

def should_continue_after_validate(state: MessagesState):
    """After validation, only proceed to agent if form is valid."""
    last_tool_msg = next(
        (m for m in reversed(state["messages"])
         if isinstance(m, ToolMessage) and m.name == "form_validate_elements"),
        None,
    )
    # Validation returns "True" on success; anything else is a failure
    if last_tool_msg and last_tool_msg.content.strip() != "True":
        return END
    return "agent"

async def run_tools_extract(state: MessagesState, config: RunnableConfig):
    tool_node_extract = ToolNode([form_get_elements])
    return await tool_node_extract.ainvoke(state, config)

async def run_tools_validate(state: MessagesState, config: RunnableConfig):
    tool_node_validate = ToolNode([form_validate_elements])
    return await tool_node_validate.ainvoke(state, config)

async def run_tools_fill(state: MessagesState, config: RunnableConfig):
    tool_node_fill = ToolNode([form_fill_fields])
    return await tool_node_fill.ainvoke(state, config)

async def run_tools_submit(state: MessagesState, config: RunnableConfig):
    tool_node_submit  = ToolNode([form_submit])
    return await tool_node_submit.ainvoke(state, config)


builder = StateGraph(MessagesState)
builder.add_node("agent",          call_agent)
builder.add_node("tools_extract",  run_tools_extract)
builder.add_node("tools_validate", run_tools_validate)
builder.add_node("tools_fill",     run_tools_fill)
builder.add_node("human_approval", human_approval_node) # retunrs Command which goto agent or END
builder.add_node("tools_submit",   run_tools_submit)

builder.add_edge(START,           "agent")
builder.add_conditional_edges(
    "agent", should_continue,
    {
        "tools_extract":  "tools_extract",
        "tools_validate": "tools_validate",
        "tools_fill":     "tools_fill",
        "tools_submit":   "tools_submit",
        END:              END,
    },
)
builder.add_edge("tools_extract",  "agent")
builder.add_conditional_edges(
    "tools_validate",
    should_continue_after_validate,
    {
        "agent": "agent",
        END: END,
    },
)
builder.add_edge("tools_fill",     "human_approval")
builder.add_edge("tools_submit",   "agent")

graph = builder.compile(checkpointer=MemorySaver())
print(graph.get_graph().draw_ascii())
#graph.get_graph().draw_png("/tmp/graph.png")

# ── Main ──────────────────────────────────────────────────────────────────────

async def main(target_url, user_info, headless=True):
    session       = BrowserSession()
    thread_config = {
        "configurable": {
            "thread_id":       "form-session-1",
            "browser_session": session,
        }
    }

    await session.start(headless=headless)
    try:
        initial_message = HumanMessage(content=(
            f"Fill out the form at {target_url} using the info below, "
            f"then ask for my approval before submitting.\n\nUser info:\n{user_info}"
        ))

        interrupted_state = None
        async for event in graph.astream(
            {"messages": [initial_message]},
            config=thread_config,
            stream_mode="values",
        ):
            last_msg = event["messages"][-1]
            if hasattr(last_msg, "content") and last_msg.content:
                print("\n[Agent]:")
                print(last_msg.content)
            snapshot = await graph.aget_state(thread_config)
            if snapshot.next and "human_approval" in snapshot.next:
                interrupted_state = snapshot
                break

        if interrupted_state is None:
            print("\nGraph finished without requiring human approval.")
            return

        # Surface the interrupt payload
        interrupt_payload = next(
            (task.interrupts[0].value
             for task in interrupted_state.tasks if task.interrupts),
            None,
        )

        print("\n" + "=" * 80)
        print("HUMAN APPROVAL REQUIRED")
        print("=" * 80)
        print(interrupt_payload.get("question", "") if interrupt_payload else "")
        print("=" * 80)
        decision = input("\nYour decision (yes/no): ").strip()

        # Resume the graph with the human decision
        async for event in graph.astream(
            Command(resume=decision),
            config=thread_config,
            stream_mode="values",
        ):
            last_msg = event["messages"][-1]
            if hasattr(last_msg, "content") and last_msg.content:
                print("\n[Agent]:")
                print(last_msg.content)

        print("\nDone.")

    finally:
        await session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Form filler agent")
    parser.add_argument("--url",       required=True, help="Target form URL")
    parser.add_argument("--user_info", required=True, help="Path to user info .txt file")
    parser.add_argument("--headless",  default=True,
                        type=lambda x: x.lower() != "false",
                        help="Headless browser? Pass --headless false to watch it run.")
    args      = parser.parse_args()
    user_info = open(args.user_info, "r", encoding="utf-8").read()
    asyncio.run(main(args.url, user_info, args.headless))


"""
Usage:
cd AgenticAI/fill_online_form
python main.py --url https://form.jotform.com/260497189942169 --user_info user_info.txt
python main.py --url https://form.jotform.com/260497189942169 --user_info user_info.txt --headless false
python main.py --url https://form.jotform.com/260497189942169 --user_info user_info2.txt
python main.py --url https://form.jotform.com/260497189942169 --user_info user_info3.txt

python main.py --url https://mendrika-alma.github.io/form-submission/ --user_info user_info4.txt --headless false

python main.py --url https://www.uber.com/careers/apply/form/153473  --user_info user_info3.txt --headless false

"""
