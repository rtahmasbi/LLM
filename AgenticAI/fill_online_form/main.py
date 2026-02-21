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

from typing import Optional
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

async def get_forms(page: Page, skip_hidden: bool = True):
    """Extract all forms and their input fields on the current page."""
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
    return form_data


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
async def form_get_elements(url: str, config: RunnableConfig) -> list:
    """
    Extract all visible form fields from a webpage.
    Returns a list of forms, each containing field metadata (id, label, type, placeholder).
    Always call this first before filling any form.
    """
    session: BrowserSession = config["configurable"]["browser_session"]
    await session.goto(url)
    await session.page.wait_for_selector("form", timeout=10000)
    return await get_forms(session.page)


@tool
async def form_fill_fields(url: str, values: dict[str, str], config: RunnableConfig) -> str:
    """
    Fill form fields on a webpage. Does NOT submit — returns a summary for human review.

    Args:
        url: The page URL.
        values: Flat mapping of field id → value,
                e.g. {"first_11": "James", "input_12": "james@email.com"}.
                Use the field ids returned by form_get_elements.
    """
    session: BrowserSession = config["configurable"]["browser_session"]
    await session.goto(url)
    summary = await _fill_page_flat(session.page, values)
    screenshot_path = "/tmp/form_preview.png"
    await session.page.screenshot(path=screenshot_path, full_page=True)
    return summary + f"\n\n[Preview screenshot saved to {screenshot_path}]"


@tool
async def form_submit(url: str, values: dict[str, str], config: RunnableConfig) -> str:
    """
    Re-fill form fields and SUBMIT the form.
    Call this ONLY after the human has approved submission.

    Args:
        url: The page URL.
        values: Same flat id→value mapping used in form_fill_fields.
    """
    session: BrowserSession = config["configurable"]["browser_session"]
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
                    "Human approved. Now call `form_submit` with "
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

all_tools        = [form_get_elements, form_fill_fields, form_submit]
tool_node_extract = ToolNode([form_get_elements])
tool_node_fill    = ToolNode([form_fill_fields])
tool_node_submit  = ToolNode([form_submit])

SYSTEM_PROMPT = """
You are a form-filling assistant. Your workflow is strictly:

1. Call `form_get_elements` to extract all visible fields from the URL.
2. Using the extracted field ids and the user info provided, build a flat
   {field_id: value} mapping and call `form_fill_fields`.
3. WAIT — a human approval step will pause execution here.
4. If approved, call `form_submit` with the SAME url and values.
5. Report the final outcome.

Never skip step 1. Never call `form_submit` before human approval.
"""


def should_continue(state: MessagesState):
    last = state["messages"][-1]
    if not hasattr(last, "tool_calls") or not last.tool_calls:
        return END
    match last.tool_calls[0]["name"]:
        case "form_get_elements": return "tools_extract"
        case "form_fill_fields":  return "tools_fill"
        case _:                   return "tools_submit"


async def call_agent(state: MessagesState):
    response = await llm.bind_tools(all_tools).ainvoke(
        [{"role": "system", "content": SYSTEM_PROMPT}] + state["messages"]
    )
    return {"messages": [response]}


async def run_tools_extract(state: MessagesState, config: RunnableConfig):
    return await tool_node_extract.ainvoke(state, config)

async def run_tools_fill(state: MessagesState, config: RunnableConfig):
    return await tool_node_fill.ainvoke(state, config)

async def run_tools_submit(state: MessagesState, config: RunnableConfig):
    return await tool_node_submit.ainvoke(state, config)


builder = StateGraph(MessagesState)
builder.add_node("agent",          call_agent)
builder.add_node("tools_extract",  run_tools_extract)
builder.add_node("tools_fill",     run_tools_fill)
builder.add_node("human_approval", human_approval_node)
builder.add_node("tools_submit",   run_tools_submit)

builder.add_edge(START,           "agent")
builder.add_conditional_edges(
    "agent", should_continue,
    {
        "tools_extract": "tools_extract",
        "tools_fill":    "tools_fill",
        "tools_submit":  "tools_submit",
        END:             END,
    },
)
builder.add_edge("tools_extract",  "agent")
builder.add_edge("tools_fill",     "human_approval")
builder.add_edge("tools_submit",   "agent")

graph = builder.compile(checkpointer=MemorySaver())
print(graph.get_graph().draw_ascii())


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(target_url, user_info):
    session       = BrowserSession()
    thread_config = {
        "configurable": {
            "thread_id":       "form-session-1",
            "browser_session": session,
        }
    }

    await session.start(headless=True)
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
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="Target form URL"
    )
    parser.add_argument(
        "--user_info",
        type=str,
        required=True,
        help="Path to user info file"
    )
    args = parser.parse_args()
    url = args.url
    user_info = open(args.user_info, "r", encoding="utf-8").read()
    asyncio.run(main(url, user_info))




"""
python AgenticAI/fill_online_form/main.py --url https://form.jotform.com/260497189942169 --user_info /home/ras/user_info.txt

"""
