
"""
This version handles the multipage
"""

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

from typing import Optional, Annotated
import operator
import argparse


# ── Extended State ─────────────────────────────────────────────────────────────

class FormState(MessagesState):
    """Extended state that accumulates fill summaries across pages."""
    fill_summaries: Annotated[list[str], operator.add]   # one entry per page
    current_page:   int                                  # 1-based page counter


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
            await self.page.goto(url, wait_until="networkidle")
            print(f"[BrowserSession] Navigated to {url}")
        else:
            print(f"[BrowserSession] Already on {url}, skipping navigation.")

    async def close(self):
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        print("[BrowserSession] Browser closed.")


# ── Helpers ───────────────────────────────────────────────────────────────────

async def get_forms(page: Page, skip_hidden: bool = True) -> dict:
    """
    Extract all forms and their input fields on the current page.
    Also detects navigation buttons (Next / Previous / Submit).
    """
    forms = await page.query_selector_all("form")
    form_data = []
    for idx, form in enumerate(forms, 1):
        inputs = await form.query_selector_all("input, select, textarea")
        fields = []
        for inp in inputs:
            if skip_hidden and not await inp.is_visible():
                continue
            field_info = {
                "label":          "",
                "type":           await inp.get_attribute("type") or "text",
                "placeholder":    await inp.get_attribute("placeholder") or "",
                "id":             await inp.get_attribute("id"),
                "name":           await inp.get_attribute("name") or "",
                "data-format":    await inp.get_attribute("data-format") or "",
                "data-seperator": await inp.get_attribute("data-seperator") or "",
                "data-maxlength": await inp.get_attribute("data-maxlength") or "",
                "is_visible":     await inp.is_visible(),
            }
            field_id = await inp.get_attribute("id")
            if field_id:
                label = await page.query_selector(f'label[for="{field_id}"]')
                if label:
                    field_info["label"] = await label.inner_text()
            fields.append(field_info)
        form_data.append({"form_index": idx, "fields": fields})

    # ── Detect page navigation buttons ────────────────────────────────────────
    # Submit button
    submit_btn = await page.query_selector(
        'button[type="submit"], input[type="submit"], '
        'button:has-text("Submit"), button:has-text("submit")'
    )
    # Next button (covers JotForm, Typeform, custom forms)
    next_btn = await page.query_selector(
        'button:has-text("Next"), input[value="Next"], '
        'button:has-text("Continue"), button:has-text("continue"), '
        'button[class*="next"], a[class*="next"]'
    )
    # Progress indicator text, e.g. "Page 2 of 4"
    progress_text = ""
    progress_el = await page.query_selector(
        '[class*="progress"], [class*="page-counter"], [class*="formPage"]'
    )
    if progress_el:
        progress_text = (await progress_el.inner_text()).strip()

    return {
        "forms":         form_data,
        "has_next":      next_btn is not None,
        "has_submit":    submit_btn is not None,
        "progress_text": progress_text,
        "current_url":   page.url,
    }


async def _fill_page_flat(page: Page, values: dict[str, str]) -> str:
    """Fill fields from a flat {id: value} dict and return a summary."""
    summary_lines = []
    for id_, user_input in values.items():
        element = await page.query_selector(f'[id="{id_}"]')
        if not element or not await element.is_visible():
            summary_lines.append(f"  [SKIP] id='{id_}' not visible or not found")
            continue
        field_type = await element.get_attribute("type") or "text"
        try:
            match field_type.lower():
                case "checkbox":
                    if user_input.strip().lower() in ("y", "yes", "true", "1"):
                        await element.check()
                    else:
                        await element.uncheck()
                case "radio":
                    await element.check()
                case "file":
                    if user_input.strip():
                        await element.set_input_files(user_input.strip())
                    else:
                        summary_lines.append(f"  [SKIP] file upload for id='{id_}'")
                        continue
                case _:
                    tag = await element.evaluate("el => el.tagName.toLowerCase()")
                    if tag == "select":
                        await element.select_option(label=user_input)
                    else:
                        await element.fill(user_input)
            summary_lines.append(f"  [OK]   id='{id_}' | value='{user_input}'")
        except Exception as e:
            summary_lines.append(f"  [ERR]  id='{id_}' — {e}")
    return "\n".join(summary_lines)


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
async def form_get_elements(url: str, config: RunnableConfig) -> dict:
    """
    Extract all visible form fields from the CURRENT page (or navigate to url first).
    Returns field metadata AND flags: has_next, has_submit, progress_text.

    Always call this:
      • at the start
      • after calling form_next_page (to get the new page's fields)
    """
    session: BrowserSession = config["configurable"]["browser_session"]
    await session.goto(url)
    try:
        await session.page.wait_for_selector("form", timeout=10_000)
    except Exception:
        pass  # some pages don't have a <form> wrapper
    return await get_forms(session.page)


@tool
async def form_fill_fields(url: str, values: dict[str, str], config: RunnableConfig) -> str:
    """
    Fill form fields on the CURRENT page. Does NOT submit or advance.
    Returns a summary for human review.

    Args:
        url:    The page URL (used only to verify we're on the right page).
        values: Flat mapping of field id → value,
                e.g. {"first_11": "James", "input_12": "james@email.com"}.
                Use the field ids returned by form_get_elements.
    """
    session: BrowserSession = config["configurable"]["browser_session"]
    # Don't re-navigate — we're already on the right page; just verify.
    if session.page.url.split("?")[0] != url.split("?")[0]:
        await session.goto(url)
    summary = await _fill_page_flat(session.page, values)
    screenshot_path = "/tmp/form_preview.png"
    await session.page.screenshot(path=screenshot_path, full_page=True)
    return summary + f"\n\n[Preview screenshot saved to {screenshot_path}]"


@tool
async def form_next_page(config: RunnableConfig) -> str:
    """
    Click the 'Next' (or 'Continue') button to advance to the next form page.
    Call form_get_elements again after this to extract the new page's fields.

    Returns the new URL or a status message.
    """
    session: BrowserSession = config["configurable"]["browser_session"]
    page = session.page

    # Try several common selectors in order of specificity
    next_selectors = [
        'button:has-text("Next")',
        'input[value="Next"]',
        'button:has-text("Continue")',
        'button[class*="next"]',
        'a[class*="next"]',
        'button:has-text("next")',
    ]
    next_btn = None
    for sel in next_selectors:
        next_btn = await page.query_selector(sel)
        if next_btn and await next_btn.is_visible():
            break
        next_btn = None

    if not next_btn:
        return "ERROR: No visible Next/Continue button found on this page."

    prev_url = page.url
    await next_btn.click()

    # Wait for either a navigation or DOM change (SPA-style forms like JotForm)
    try:
        await page.wait_for_load_state("networkidle", timeout=8_000)
    except Exception:
        await asyncio.sleep(1)  # fallback for SPA transitions

    new_url = page.url
    screenshot_path = "/tmp/next_page.png"
    await page.screenshot(path=screenshot_path, full_page=True)

    if new_url != prev_url:
        return f"Advanced to next page. New URL: {new_url}"
    else:
        return (
            f"Clicked Next (SPA transition — URL unchanged: {new_url}). "
            "New fields should now be visible. Call form_get_elements to re-extract."
        )


@tool
async def form_submit(url: str, config: RunnableConfig) -> str:
    """
    Submit the form on the CURRENT page (last page only).
    Call this ONLY after the human has approved submission.
    Do NOT re-fill fields — they are already filled.
    """
    session: BrowserSession = config["configurable"]["browser_session"]
    page = session.page

    submit_btn = await page.query_selector(
        'button[type="submit"], input[type="submit"], '
        'button:has-text("Submit"), button:has-text("submit")'
    )
    if submit_btn and await submit_btn.is_visible():
        await submit_btn.click()
    else:
        await page.keyboard.press("Enter")

    try:
        await page.wait_for_load_state("networkidle", timeout=10_000)
    except Exception:
        await asyncio.sleep(2)

    screenshot_path = "/tmp/final_state.png"
    await page.screenshot(path=screenshot_path, full_page=True)
    print(f"[form_submit] Final screenshot saved to {screenshot_path}")
    return f"Form submitted successfully. Final URL: {page.url}"


# ── Human approval middleware ─────────────────────────────────────────────────

async def human_approval_node(state: FormState) -> Command:
    """
    Interrupt execution and show the human a summary of ALL pages filled so far.
    Resume with 'yes' → submit, 'no' → cancel.
    """
    all_summaries = state.get("fill_summaries", [])
    combined = "\n\n".join(
        f"--- Page {i+1} ---\n{s}" for i, s in enumerate(all_summaries)
    )
    if not combined:
        # Fallback: pull from the last ToolMessage
        combined = next(
            (m.content for m in reversed(state["messages"])
             if isinstance(m, ToolMessage) and "form_fill_fields" in (m.name or "")),
            "No fill summary found.",
        )

    human_decision = interrupt({
        "question": (
            "The multi-page form has been filled. Please review all pages below "
            "and decide whether to submit.\n\n"
            f"{combined}\n\n"
            "Reply 'yes' to submit or 'no' to cancel."
        )
    })

    if str(human_decision).strip().lower() in ("yes", "y"):
        return Command(
            goto="agent",
            update={
                "messages": state["messages"] + [
                    HumanMessage(content=(
                        "Human approved. Now call `form_submit` to submit the form. "
                        "Do NOT re-fill any fields — they are already on the page."
                    ))
                ]
            },
        )
    else:
        return Command(
            goto=END,
            update={
                "messages": state["messages"] + [
                    HumanMessage(content="User declined. Task cancelled.")
                ]
            },
        )


# ── Agent node ────────────────────────────────────────────────────────────────

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

all_tools = [form_get_elements, form_fill_fields, form_next_page, form_submit]

SYSTEM_PROMPT = """
You are a multi-page form-filling assistant. Your strict workflow:

LOOP (repeat for every page until a Submit button is found):
  1. Call `form_get_elements` with the target URL to get visible fields,
     plus `has_next`, `has_submit`, and `progress_text` flags.
  2. Using the user info and the extracted field ids, build a flat
     {field_id: value} mapping for THIS PAGE ONLY and call `form_fill_fields`.
  3a. If `has_next` is True  → call `form_next_page`, then go back to step 1.
  3b. If `has_submit` is True → STOP. Do NOT call form_submit yet.
      A human approval step will automatically pause execution.

AFTER APPROVAL:
  4. Call `form_submit` (no arguments for values — fields are already filled).
  5. Report the final outcome.

Rules:
- Never call `form_submit` before the human approves.
- Never skip `form_get_elements` when arriving on a new page.
- If a field in the user info has no matching id on the current page, skip it
  (it likely belongs to a different page).
- If `has_next` and `has_submit` are both True, prefer `has_next` (not the last page yet).
"""


async def call_agent(state: FormState):
    response = await llm.bind_tools(all_tools).ainvoke(
        [{"role": "system", "content": SYSTEM_PROMPT}] + state["messages"]
    )
    return {"messages": [response]}


# ── Tool nodes with state side-effects ────────────────────────────────────────

async def run_tools_extract(state: FormState, config: RunnableConfig):
    return await ToolNode([form_get_elements]).ainvoke(state, config)

async def run_tools_fill(state: FormState, config: RunnableConfig):
    result = await ToolNode([form_fill_fields]).ainvoke(state, config)
    # Accumulate the fill summary into state
    last_tool_msg = result["messages"][-1]
    page_num = state.get("current_page", 1)
    return {
        **result,
        "fill_summaries": [last_tool_msg.content],   # Annotated[list, add] appends
        "current_page":   page_num + 1,
    }

async def run_tools_next(state: FormState, config: RunnableConfig):
    return await ToolNode([form_next_page]).ainvoke(state, config)

async def run_tools_submit(state: FormState, config: RunnableConfig):
    return await ToolNode([form_submit]).ainvoke(state, config)


# ── Router ─────────────────────────────────────────────────────────────────────

def should_continue(state: FormState):
    last = state["messages"][-1]
    if not hasattr(last, "tool_calls") or not last.tool_calls:
        return END
    match last.tool_calls[0]["name"]:
        case "form_get_elements": return "tools_extract"
        case "form_fill_fields":  return "tools_fill"
        case "form_next_page":    return "tools_next"
        case _:                   return "tools_submit"


# ── Graph ─────────────────────────────────────────────────────────────────────

builder = StateGraph(FormState)

builder.add_node("agent",          call_agent)
builder.add_node("tools_extract",  run_tools_extract)
builder.add_node("tools_fill",     run_tools_fill)
builder.add_node("tools_next",     run_tools_next)
builder.add_node("human_approval", human_approval_node)
builder.add_node("tools_submit",   run_tools_submit)

builder.add_edge(START,           "agent")
builder.add_conditional_edges(
    "agent", should_continue,
    {
        "tools_extract": "tools_extract",
        "tools_fill":    "tools_fill",
        "tools_next":    "tools_next",
        "tools_submit":  "tools_submit",
        END:             END,
    },
)
builder.add_edge("tools_extract",  "agent")   # always back to agent after extraction
builder.add_edge("tools_fill",     "agent")   # agent decides: next page or done?
builder.add_edge("tools_next",     "agent")   # agent re-extracts after advancing
builder.add_edge("tools_submit",   "agent")   # agent reports final outcome

# ── Human approval wiring ─────────────────────────────────────────────────────
# The agent signals "ready for approval" by producing a message with no tool
# calls AND fill_summaries being non-empty. We intercept that via a custom
# router added after tools_fill.

def route_after_fill(state: FormState):
    """
    After filling a page the agent decides what to do next.
    If the agent's last message has no tool call but we have summaries,
    it means it's signalling 'I'm done filling all pages — approve me'.
    """
    last = state["messages"][-1]
    # If the last AI message calls a tool, route normally
    if hasattr(last, "tool_calls") and last.tool_calls:
        return should_continue(state)
    # No tool call → agent is done filling; go to human approval
    if state.get("fill_summaries"):
        return "human_approval"
    return END

# Re-wire: after tools_fill go through the router instead of directly to agent
builder.add_edge("tools_fill", "agent")

graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["human_approval"],   # pause before running human_approval
)
print(graph.get_graph().draw_ascii())


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(target_url: str, user_info: str, headless:bool):
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
            f"Fill out the (possibly multi-page) form at {target_url} using the "
            f"info below, then ask for my approval before submitting.\n\nUser info:\n{user_info}"
        ))

        print("\n[Main] Starting form-filling agent...\n")

        # ── Stream until interrupted ───────────────────────────────────────────
        interrupted_state = None
        async for event in graph.astream(
            {"messages": [initial_message], "fill_summaries": [], "current_page": 1},
            config=thread_config,
            stream_mode="values",
        ):
            last_msg = event["messages"][-1]
            if hasattr(last_msg, "content") and last_msg.content:
                print(f"\n[Agent]: {last_msg.content}")

            snapshot = await graph.aget_state(thread_config)
            if snapshot.next and "human_approval" in snapshot.next:
                interrupted_state = snapshot
                break

        if interrupted_state is None:
            print("\n[Main] Graph finished without requiring human approval.")
            return

        # ── Build approval summary from accumulated state ──────────────────────
        summaries = interrupted_state.values.get("fill_summaries", [])
        combined  = "\n\n".join(
            f"--- Page {i+1} ---\n{s}" for i, s in enumerate(summaries)
        ) or "No fill summaries captured."

        print("\n" + "=" * 80)
        print("HUMAN APPROVAL REQUIRED — Review all filled pages:")
        print("=" * 80)
        print(combined)
        print("=" * 80)
        decision = input("\nSubmit the form? (yes/no): ").strip()

        # ── Resume with human decision ─────────────────────────────────────────
        async for event in graph.astream(
            Command(resume=decision),
            config=thread_config,
            stream_mode="values",
        ):
            last_msg = event["messages"][-1]
            if hasattr(last_msg, "content") and last_msg.content:
                print(f"\n[Agent]: {last_msg.content}")

        print("\n[Main] Done.")

    finally:
        await session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-page form filler agent")
    parser.add_argument("--url",       required=True, help="Target form URL")
    parser.add_argument("--user_info", required=True, help="Path to user info file")
    parser.add_argument("--headless",  dafault=True, help="if the page is headless")
    args     = parser.parse_args()
    user_info = open(args.user_info, "r", encoding="utf-8").read()
    asyncio.run(main(args.url, user_info, args.headless))


"""
Usage:
    python main.py --url https://form.jotform.com/260497189942169 --user_info /home/user/info.txt


"""


