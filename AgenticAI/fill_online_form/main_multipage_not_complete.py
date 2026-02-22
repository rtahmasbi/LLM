import json
import asyncio
from playwright.async_api import async_playwright, Page, Browser

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
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
    fill_summaries: Annotated[list[str], operator.add]  # accumulates one entry per page
    current_page:   int                                  # 1-based


# ── Shared browser session ────────────────────────────────────────────────────

class BrowserSession:
    def __init__(self):
        self._playwright = None
        self._browser: Optional[Browser] = None
        self.page: Optional[Page] = None

    async def start(self, headless: bool = True):
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=headless)
        self.page = await self._browser.new_page()
        print("[BrowserSession] Browser started.")

    async def ensure_on(self, url: str):
        """Navigate to url only on first visit. SPA forms never change URL after
        Next clicks, so we must NOT re-navigate — we just read the current DOM."""
        current = self.page.url if self.page else ""
        if current in ("about:blank", "") or not current.startswith("http"):
            await self.page.goto(url, wait_until="networkidle")
            print(f"[BrowserSession] Navigated to {url}")
        else:
            print(f"[BrowserSession] On '{current}' — reading DOM without re-navigating.")

    async def close(self):
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        print("[BrowserSession] Browser closed.")


# ── DOM helpers ───────────────────────────────────────────────────────────────

async def get_forms(page: Page, skip_hidden: bool = True) -> dict:
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
                label_el = await page.query_selector(f'label[for="{field_id}"]')
                if label_el:
                    field_info["label"] = await label_el.inner_text()
            fields.append(field_info)
        form_data.append({"form_index": idx, "fields": fields})

    submit_btn = await page.query_selector(
        'button[type="submit"], input[type="submit"], '
        'button:has-text("Submit"), button:has-text("submit")'
    )
    next_btn = await page.query_selector(
        'button:has-text("Next"), input[value="Next"], '
        'button:has-text("Continue"), button:has-text("continue"), '
        'button[class*="next"], a[class*="next"]'
    )
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
    """Extract all visible form fields from the current page.
    On the first call it navigates to url; afterwards it reads the live DOM
    without re-navigating (required for SPA forms like JotForm).
    Returns field metadata plus has_next, has_submit, progress_text.
    ALWAYS call this after form_next_page to see the new page fields."""
    session: BrowserSession = config["configurable"]["browser_session"]
    await session.ensure_on(url)
    await asyncio.sleep(0.6)   # let SPA animations settle
    try:
        await session.page.wait_for_selector("form", timeout=10_000)
    except Exception:
        pass
    result = await get_forms(session.page)
    print(f"[form_get_elements] has_next={result['has_next']}  "
          f"has_submit={result['has_submit']}  progress='{result['progress_text']}'")
    return result


@tool
async def form_fill_fields(url: str, values: dict[str, str], config: RunnableConfig) -> str:
    """Fill form fields on the CURRENT page. Does NOT advance or submit.
    Args:
        url: The page URL (used for first-time navigation only).
        values: Flat mapping of field id to value, from form_get_elements."""
    session: BrowserSession = config["configurable"]["browser_session"]
    await session.ensure_on(url)
    summary = await _fill_page_flat(session.page, values)
    screenshot_path = "/tmp/form_preview.png"
    await session.page.screenshot(path=screenshot_path, full_page=True)
    print(f"[form_fill_fields] Screenshot: {screenshot_path}")
    return summary + f"\n\n[Preview screenshot: {screenshot_path}]"


@tool
async def form_next_page(config: RunnableConfig) -> str:
    """Click the Next or Continue button to advance the form to the next page.
    Works for both URL-based and SPA (single-page) forms.
    ALWAYS call form_get_elements after this to re-extract the new page fields."""
    session: BrowserSession = config["configurable"]["browser_session"]
    page = session.page

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
        candidate = await page.query_selector(sel)
        if candidate and await candidate.is_visible():
            next_btn = candidate
            break

    if not next_btn:
        return "ERROR: No visible Next/Continue button found on this page."

    prev_url = page.url
    await next_btn.click()

    try:
        await page.wait_for_load_state("networkidle", timeout=6_000)
    except Exception:
        pass
    await asyncio.sleep(0.8)   # extra settle for SPA animations

    new_url = page.url
    await page.screenshot(path="/tmp/next_page.png", full_page=True)
    print("[form_next_page] Screenshot saved to /tmp/next_page.png")

    if new_url != prev_url:
        return f"Navigated to next page. New URL: {new_url}"
    return (
        f"Clicked Next — SPA transition, URL unchanged ({new_url}). "
        "New fields should now be visible. Call form_get_elements to re-extract."
    )


@tool
async def form_submit(config: RunnableConfig) -> str:
    """Submit the form on the current (last) page. Call ONLY after human approval.
    Fields are already filled — do NOT pass or re-fill values."""
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

    await page.screenshot(path="/tmp/final_state.png", full_page=True)
    print("[form_submit] Final screenshot saved to /tmp/final_state.png")
    return f"Form submitted. Final URL: {page.url}"


# ── Human approval node ───────────────────────────────────────────────────────

async def human_approval_node(state: FormState) -> Command:
    all_summaries = state.get("fill_summaries", [])
    combined = "\n\n".join(
        f"--- Page {i+1} ---\n{s}" for i, s in enumerate(all_summaries)
    ) or "No fill summaries captured."

    human_decision = interrupt({
        "question": (
            "Multi-page form filled. Please review:\n\n"
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
                        "Human approved. Call form_submit now. "
                        "Fields are already filled — do not re-fill them."
                    ))
                ]
            },
        )
    return Command(
        goto=END,
        update={
            "messages": state["messages"] + [
                HumanMessage(content="User declined. Task cancelled.")
            ]
        },
    )


# ── LLM + system prompt ───────────────────────────────────────────────────────

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
all_tools = [form_get_elements, form_fill_fields, form_next_page, form_submit]

SYSTEM_PROMPT = """
You are a multi-page form-filling assistant. Follow this workflow exactly:

LOOP — repeat for every page until you reach the Submit button:
  1. Call form_get_elements with the target URL.
     Check has_next, has_submit, and progress_text.
  2. Build a flat {field_id: value} dict for fields visible on THIS page only,
     then call form_fill_fields.
  3a. If has_next is True  → call form_next_page, then go back to step 1.
  3b. If has_submit is True (and has_next is False) → write:
      "All pages filled. Waiting for human approval." and STOP.
      Do NOT call form_submit — approval is handled automatically.

AFTER HUMAN APPROVES:
  4. Call form_submit (no arguments needed).
  5. Report the final URL.

Rules:
- NEVER call form_submit before human approval.
- ALWAYS call form_get_elements after form_next_page.
- Skip user-info fields that have no matching id on the current page.
- If both has_next and has_submit are True, treat it as not the last page (prefer has_next).
- Only use field ids returned by form_get_elements — never invent them.
"""


# ── Agent node ────────────────────────────────────────────────────────────────

async def call_agent(state: FormState):
    response = await llm.bind_tools(all_tools).ainvoke(
        [{"role": "system", "content": SYSTEM_PROMPT}] + state["messages"]
    )
    return {"messages": [response]}


# ── Tool runner nodes ─────────────────────────────────────────────────────────

async def run_tools_extract(state: FormState, config: RunnableConfig):
    return await ToolNode([form_get_elements]).ainvoke(state, config)

async def run_tools_fill(state: FormState, config: RunnableConfig):
    result = await ToolNode([form_fill_fields]).ainvoke(state, config)
    last_tool_msg = result["messages"][-1]
    page_num = state.get("current_page", 1)
    return {
        **result,
        "fill_summaries": [last_tool_msg.content],  # operator.add appends to list
        "current_page":   page_num + 1,
    }

async def run_tools_next(state: FormState, config: RunnableConfig):
    return await ToolNode([form_next_page]).ainvoke(state, config)

async def run_tools_submit(state: FormState, config: RunnableConfig):
    return await ToolNode([form_submit]).ainvoke(state, config)


# ── Router ────────────────────────────────────────────────────────────────────

def route_agent(state: FormState):
    last = state["messages"][-1]

    if hasattr(last, "tool_calls") and last.tool_calls:
        match last.tool_calls[0]["name"]:
            case "form_get_elements": return "tools_extract"
            case "form_fill_fields":  return "tools_fill"
            case "form_next_page":    return "tools_next"
            case "form_submit":       return "tools_submit"
            case _:                   return END

    # No tool call AND we have summaries → agent finished all pages, go to approval
    if state.get("fill_summaries"):
        return "human_approval"

    return END


# ── Graph ─────────────────────────────────────────────────────────────────────

builder = StateGraph(FormState)
builder.add_node("agent",          call_agent)
builder.add_node("tools_extract",  run_tools_extract)
builder.add_node("tools_fill",     run_tools_fill)
builder.add_node("tools_next",     run_tools_next)
builder.add_node("human_approval", human_approval_node)
builder.add_node("tools_submit",   run_tools_submit)

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent", route_agent,
    {
        "tools_extract":  "tools_extract",
        "tools_fill":     "tools_fill",
        "tools_next":     "tools_next",
        "tools_submit":   "tools_submit",
        "human_approval": "human_approval",
        END:              END,
    },
)
builder.add_edge("tools_extract", "agent")
builder.add_edge("tools_fill",    "agent")
builder.add_edge("tools_next",    "agent")
builder.add_edge("tools_submit",  "agent")
# human_approval uses Command(goto=...) — no static outgoing edge needed

graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["human_approval"],
)
print(graph.get_graph().draw_ascii())


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(target_url: str, user_info: str, headless: bool = True):
    session = BrowserSession()
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

        interrupted_state = None
        async for event in graph.astream(
            {"messages": [initial_message], "fill_summaries": [], "current_page": 1},
            config=thread_config,
            stream_mode="values",
        ):
            last_msg = event["messages"][-1]
            if hasattr(last_msg, "content") and last_msg.content:
                role = "Agent" if isinstance(last_msg, AIMessage) else "System"
                print(f"\n[{role}]: {last_msg.content}")

            snapshot = await graph.aget_state(thread_config)
            if snapshot.next and "human_approval" in snapshot.next:
                interrupted_state = snapshot
                break

        if interrupted_state is None:
            print("\n[Main] Graph finished without pausing for human approval.")
            return

        # Build and display the full multi-page fill summary
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

        # Resume the graph with the human's decision
        async for event in graph.astream(
            Command(resume=decision),
            config=thread_config,
            stream_mode="values",
        ):
            last_msg = event["messages"][-1]
            if hasattr(last_msg, "content") and last_msg.content:
                role = "Agent" if isinstance(last_msg, AIMessage) else "System"
                print(f"\n[{role}]: {last_msg.content}")

        print("\n[Main] Done.")

    finally:
        await session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-page form filler agent")
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
python main.py --url https://form.jotform.com/260497189942169 --user_info user_info.txt
python main.py --url https://form.jotform.com/260497189942169 --user_info user_info.txt --headless false

"""
