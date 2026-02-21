import os
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

from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


user_info = """
name: James
family name: Abdi
phone number: +1222 333 4444
email: james.a@gmail.com
Position of interest: data science jobs
"""


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


# ── Pydantic models ───────────────────────────────────────────────────────────

class FieldElement(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    label: str
    type: str
    placeholder: str
    id: Optional[str]
    name: str
    data_format: str = Field(alias="data-format")
    data_seperator: str = Field(alias="data-seperator")
    data_maxlength: str = Field(alias="data-maxlength")
    validate: str
    is_visible: bool
    value: str


class FormItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    form_index: int = Field(..., description="Index of the form; 1-based position.")
    fields: List[FieldElement]


class FormElements(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: List[FormItem]


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
                "required":       await inp.get_attribute("required") is not None,
                "id":             await inp.get_attribute("id"),
                "name":           await inp.get_attribute("name") or "",
                "data-format":    await inp.get_attribute("data-format") or "",
                "data-seperator": await inp.get_attribute("data-seperator") or "",
                "data-maxlength": await inp.get_attribute("data-maxlength") or "",
                "validate":       await inp.get_attribute("class") or "",
                "is_visible":     await inp.is_visible(),
                "value":          "",
            }
            field_id = await inp.get_attribute("id")
            if field_id:
                label = await page.query_selector(f'label[for="{field_id}"]')
                if label:
                    field_info["label"] = await label.inner_text()
            fields.append(field_info)
        form_data.append({"form_index": idx, "fields": fields})
    return form_data


async def _fill_page(page: Page, form_elements: FormElements) -> str:
    """Fill all visible fields and return a human-readable summary."""
    summary_lines = []
    for form_item in form_elements.items:
        summary_lines.append(f"\n=== FORM {form_item.form_index} ===")
        for field in form_item.fields:
            if field.type == "hidden":
                continue
            id_ = field.id
            if not id_:
                summary_lines.append(f"  [SKIP] No id for field '{field.label}'")
                continue
            element = await page.query_selector(f'[id="{id_}"]')
            if not element or not await element.is_visible():
                summary_lines.append(f"  [SKIP] id='{id_}' not visible")
                continue
            user_input = field.value
            try:
                match field.type.lower():
                    case "checkbox":
                        if user_input.strip().lower() in ("y", "yes", "true", "1"):
                            await element.check()
                        else:
                            await element.uncheck()
                    case "radio":
                        await element.check()
                    case "select":
                        await element.select_option(label=user_input)
                    case "file":
                        if user_input.strip():
                            await element.set_input_files(user_input.strip())
                        else:
                            summary_lines.append(f"  [SKIP] file upload for id='{id_}'")
                            continue
                    case _:
                        await element.fill(user_input)
                summary_lines.append(
                    f"  [OK]   id='{id_}' | label='{field.label}' | value='{user_input}'"
                )
            except Exception as e:
                summary_lines.append(f"  [ERR]  id='{id_}' — {e}")
    return "\n".join(summary_lines)


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
async def form_get_elements(url: str, config: RunnableConfig) -> list:
    """Extract all form elements from a webpage using the shared browser session."""
    session: BrowserSession = config["configurable"]["browser_session"]
    await session.goto(url)
    await session.page.wait_for_selector("form", timeout=10000)
    return await get_forms(session.page)


@tool
async def form_fill_fields(url: str, filled_form: str, config: RunnableConfig) -> str:
    """
    Fill all form fields using the provided pre-filled JSON string (FormElements schema).
    Does NOT submit — returns a summary for human review.
    """
    session: BrowserSession = config["configurable"]["browser_session"]
    await session.goto(url)
    form_elements = FormElements.model_validate_json(filled_form)
    summary = await _fill_page(session.page, form_elements)
    screenshot_path = "/tmp/form_preview.png"
    await session.page.screenshot(path=screenshot_path, full_page=True)
    return summary + f"\n\n[Preview screenshot saved to {screenshot_path}]"


@tool
async def form_submit(url: str, filled_form: str, config: RunnableConfig) -> str:
    """
    Re-fill all form fields and SUBMIT the form.
    Call this ONLY after the human has approved submission.
    """
    session: BrowserSession = config["configurable"]["browser_session"]
    await session.goto(url)
    form_elements = FormElements.model_validate_json(filled_form)
    await _fill_page(session.page, form_elements)

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

    # Find the original form_fill_fields tool-call args for re-use in form_submit
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
                    f"filled_form=\"{fill_args.get('filled_form', '')}\"."
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


# ── Graph ─────────────────────────────────────────────────────────────────────

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

all_tools        = [form_get_elements, form_fill_fields, form_submit]
tool_node_extract = ToolNode([form_get_elements])
tool_node_fill    = ToolNode([form_fill_fields])
tool_node_submit  = ToolNode([form_submit])

SYSTEM_PROMPT = """
You are a form-filling assistant. Your workflow is strictly:

1. Call `form_get_elements` to extract the form fields from the URL.
2. Using the extracted fields AND the user info provided, build a filled JSON
   that matches the FormElements schema, then call `form_fill_fields`.
3. WAIT — a human approval step will pause execution here.
4. If approved, call `form_submit` with the SAME url and filled_form arguments.
5. Report the final outcome.

Never call `form_submit` before human approval.
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

async def main():
    target_url    = "https://form.jotform.com/260497189942169"
    session       = BrowserSession()
    thread_config = {
        "configurable": {
            "thread_id":       "form-session-1",
            "browser_session": session,
        }
    }

    await session.start(headless=True)
    try:
        # Single entry point — the agent handles extract → fill → approve → submit
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
                print(f"\n[Agent]: {last_msg.content}")
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

        # Resume
        async for event in graph.astream(
            Command(resume=decision),
            config=thread_config,
            stream_mode="values",
        ):
            last_msg = event["messages"][-1]
            if hasattr(last_msg, "content") and last_msg.content:
                print(f"\n[Agent]: {last_msg.content}")

        print("\nDone.")

    finally:
        await session.close()


if __name__ == "__main__":
    asyncio.run(main())




"""
python AgenticAI/fill_online_form/main.py

"""
