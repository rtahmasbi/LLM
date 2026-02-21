
import os
import json
import asyncio
from playwright.async_api import async_playwright
import time

import argparse
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
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



class FieldElement(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    label: str
    type: str
    placeholder: str
    id: Optional[str]  # anyOf string or null
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


def load_and_index_pdf(pdf_path: str):
    """Load PDF with PyPDF and index it into a FAISS vector store."""
    print(f"  Loading PDF: {pdf_path}")
    # Load PDF using langchain-community PyPDFLoader (uses pypdf under the hood)
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"   Loaded {len(documents)} page(s).")
    # Split documents into manageable chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(documents)
    print(f"   Split into {len(chunks)} chunk(s).")
    # Embed and store in FAISS
    print("   Creating vector store...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("   Vector store ready.\n")
    return vectorstore


def load_and_index_text(str_text: str):
    texts = [str_text]
    # Embed and store in FAISS
    print("   Creating vector store...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts, embeddings)
    print("   Vector store ready.\n")
    return vectorstore


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
                "value": "",
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



@tool
async def form_get_elements(url: str, headless: bool = True) -> list:
    """Extract all form elements from a webpage."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        page = await browser.new_page()
        await page.goto(url, wait_until="networkidle")
        await page.wait_for_selector("form", timeout=10000)
        forms = await get_forms(page)
        await browser.close()
        return forms




async def fill_forms(ret_llm, url, headless=True):
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




@tool
async def form_fill_fields(url: str, filled_form: str, headless: bool = True) -> str:
    """
    Fill all form fields on a webpage using the provided pre-filled JSON (as a string).
    Does NOT submit — returns a summary of filled fields for human review.

    Args:
        url: The URL of the page containing the form.
        filled_form: JSON string matching the FormElements schema with `value` populated.
        headless: Whether to run the browser in headless mode.

    Returns:
        A human-readable summary of which fields were filled.
    """
    form_elements: FormElements = FormElements.model_validate_json(filled_form)
    summary_lines = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        page = await browser.new_page()
        await page.goto(url, wait_until="networkidle")
        
        for form_item in form_elements.items:
            summary_lines.append(f"\n=== FORM {form_item.form_index} ===")
            for field in form_item.fields:
                if field.type == "hidden":
                    continue
                id_ = field.id
                if not id_:
                    summary_lines.append(f"  [SKIP] No id for field '{field.label}'")
                    continue
                selector = f'[id="{id_}"]'
                element = await page.query_selector(selector)
                if not element or not await element.is_visible():
                    summary_lines.append(f"  [SKIP] id='{id_}' not visible")
                    continue
                
                user_input = field.value
                field_type = field.type.lower()
                try:
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
                            summary_lines.append(f"  [SKIP] file upload for id='{id_}'")
                            continue
                    else:
                        await element.fill(user_input)
                    summary_lines.append(f"  [OK]   id='{id_}' | label='{field.label}' | value='{user_input}'")
                except Exception as e:
                    summary_lines.append(f"  [ERR]  id='{id_}' — {e}")
        
        # Store page state for later submission (screenshot as confirmation)
        screenshot_path = "/tmp/form_preview.png"
        await page.screenshot(path=screenshot_path, full_page=True)
        await browser.close()
    #
    summary = "\n".join(summary_lines)
    summary += f"\n\n[Preview screenshot saved to {screenshot_path}]"
    return summary


@tool
async def form_submit(url: str, filled_form: str, headless: bool = True) -> str:
    """
    Re-fill all form fields and SUBMIT the form.
    Call this ONLY after the human has approved submission.

    Args:
        url: The URL of the page containing the form.
        filled_form: JSON string matching the FormElements schema with `value` populated.
        headless: Whether to run the browser in headless mode.

    Returns:
        Confirmation message with final URL after submission.
    """
    form_elements: FormElements = FormElements.model_validate_json(filled_form)
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        page = await browser.new_page()
        await page.goto(url, wait_until="networkidle")

        for form_item in form_elements.items:
            for field in form_item.fields:
                if field.type == "hidden":
                    continue
                id_ = field.id
                if not id_:
                    continue
                selector = f'[id="{id_}"]'
                element = await page.query_selector(selector)
                if not element or not await element.is_visible():
                    continue
                user_input = field.value
                field_type = field.type.lower()
                try:
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
                        await element.fill(user_input)
                except Exception:
                    pass

            # Submit the form
            submit_btn = await page.query_selector(
                'form button[type="submit"], form input[type="submit"]'
            )
            if submit_btn:
                await submit_btn.click()
                await page.wait_for_load_state("networkidle")
            else:
                # fallback: press Enter on last filled element
                await page.keyboard.press("Enter")
                await page.wait_for_load_state("networkidle")

        final_url = page.url
        await browser.close()
        return f"Form submitted successfully. Final URL: {final_url}"


################################################################ middleware
def human_approval_node(state: dict) -> Command:
    """
    Middleware node that pauses graph execution and asks the human whether to
    proceed with form submission.  Uses LangGraph's `interrupt()` primitive so
    the graph can be resumed later via `graph.invoke(Command(resume=...))`.

    The node inspects the last AI message for a tool call to `form_fill_fields`
    and surfaces the fill summary to the operator.
    """
    last_message = state["messages"][-1]
    # Collect the fill summary from the most recent ToolMessage
    fill_summary = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage) and "form_fill_fields" in (msg.name or ""):
            fill_summary = msg.content
            break
    # `interrupt()` pauses the graph and returns a value when resumed.
    human_decision = interrupt(
        {
            "question": (
                "The form has been filled. Please review the field values below "
                "and decide whether to submit.\n\n"
                f"{fill_summary}\n\n"
                "Reply 'yes' to submit or 'no' to cancel."
            )
        }
    )
    # Route based on human response
    if str(human_decision).strip().lower() in ("yes", "y"):
        return Command(goto="agent")  # resume agent so it can call form_submit
    else:
        # Inject a message telling the agent the user declined submission
        return Command(
            goto="__end__",
            update={
                "messages": state["messages"]
                + [HumanMessage(content="User declined form submission. Task cancelled.")]
            },
        )



#########################################################################
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm = llm.with_structured_output(FormElements)

# Tools for the page-extraction agent (unchanged)
extraction_tools = [form_get_elements]

# Tools for the fill-forms agent
tool_node_fill   = ToolNode([form_fill_fields])
tool_node_submit = ToolNode([form_submit])

# MemorySaver enables graph checkpointing required for interrupt/resume
memory = MemorySaver()

FILL_FORMS_SYSTEM_PROMPT = """
You are a form-filling assistant. Your workflow is strictly:

1. Call `form_fill_fields` with the URL and the pre-filled JSON to populate every
   visible field and get a summary for human review.
2. WAIT — a human approval middleware will pause execution at this point.
3. If the human approves, call `form_submit` with the SAME url and filled_form
   arguments to re-fill and submit the form.
4. Report the final outcome to the user.

Never call `form_submit` before `form_fill_fields` has been reviewed by a human.
"""

# Build the fill-forms agent graph with the human-approval middleware injected
# between the agent and the tools.  We use create_react_agent and then add
# a custom node into the compiled graph.


def should_continue(state: MessagesState):
    """Route: after the agent responds, decide what to do next."""
    last = state["messages"][-1]
    if not hasattr(last, "tool_calls") or not last.tool_calls:
        return END
    # If the agent wants to call form_fill_fields → go to tools then human check
    tool_name = last.tool_calls[0]["name"]
    if tool_name == "form_fill_fields":
        return "tools_fill"
    # If the agent wants to call form_submit → go to tools directly (already approved)
    return "tools_submit"


async def call_agent(state: MessagesState):
    """Async agent node — avoids sync/async mismatch with Playwright tools."""
    bound_llm = llm.bind_tools(fill_tools)
    response = await bound_llm.ainvoke(
        [{"role": "system", "content": FILL_FORMS_SYSTEM_PROMPT}] + state["messages"]
    )
    return {"messages": [response]}

async def run_tools_fill(state: MessagesState):
    """Async wrapper around the fill ToolNode."""
    return await tool_node_fill.ainvoke(state)


async def run_tools_submit(state: MessagesState):
    """Async wrapper around the submit ToolNode."""
    return await tool_node_submit.ainvoke(state)



# Build the graph
builder = StateGraph(MessagesState)

builder.add_node("agent", call_agent)
builder.add_node("tools_fill", tool_node)          # fills without submitting
builder.add_node("human_approval", human_approval_node)  # pause for review
builder.add_node("tools_submit", tool_node)         # actual submission

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools_fill":   "tools_fill",
        "tools_submit": "tools_submit",
        END:            END,
    },
)
# After filling → pause for human
builder.add_edge("tools_fill", "human_approval")
# human_approval uses Command(goto=...) so no explicit edge needed here
# After submission tool → back to agent to report result
builder.add_edge("tools_submit", "agent")

fill_forms_graph = builder.compile(checkpointer=memory, interrupt_before=[])

#########################################################################

agent_page_elements = create_agent(
    llm,
    extraction_tools,
    system_prompt="You are a helpful assistant that uses tools to answer questions.",
)

prompt_llm = """
You are given some user information and your task is to fill the values of each element in the given json object.
- Check the `label` carefully and make your answer based on it
- For each element, fill the `value` with the right answer
- For each `value`, make sure to follow the `placeholder` or `data_format` format.
- If you dont have any information about the element, just fill it with blank.
- For the phone number, convert it to the `placeholder` format.
- Make sure the return json object is valid.
"""




async def main():
    target_url = "https://form.jotform.com/260497189942169"
    thread_config = {"configurable": {"thread_id": "form-session-1"}}

    # ── Step 1: Extract form elements ──────────────────────────────────────────
    print("-"*80, "Step 1: Extracting form elements...")
    response = await agent_page_elements.ainvoke(
        {"messages": [{"role": "user", "content": f"extract the form elements from {target_url}"}]}
    )
    jsn_elements = {}
    for msg in reversed(response["messages"]):
        if msg.type == "tool":
            jsn_elements = json.loads(msg.content)
            break
    print(json.dumps(jsn_elements, indent=4))

    # ── Step 2: LLM fills in the values ───────────────────────────────────────
    print("\nStep 2: Filling field values with LLM...")
    content = (
        f"Here is the JSON elements:\n{json.dumps(jsn_elements, indent=2)}\n\n"
        f"user_info:\n{user_info}"
    )
    # ainvoke keeps us fully in async context (no sync blocking)
    filled: FormElements = await structured_llm.ainvoke(content)
    filled_json_str = filled.model_dump_json(by_alias=True)
    print(filled.model_dump_json(indent=2))

    # ── Step 3: Run the fill-forms agent (will pause for approval) ────────────
    print("\nStep 3: Running fill-forms agent (will pause for human approval)...")
    user_message = (
        f"Please fill the form at {target_url} using this pre-filled JSON:\n{filled_json_str}"
    )

    # Use astream so Playwright's async event loop is never blocked by sync calls.
    interrupted_state = None
    async for event in fill_forms_graph.astream(
        {"messages": [HumanMessage(content=user_message)]},
        config=thread_config,
        stream_mode="values",
    ):
        last_msg = event["messages"][-1]
        if hasattr(last_msg, "content") and last_msg.content:
            print(f"\n[Agent]: {last_msg.content}")
        # Check whether the graph has paused inside human_approval
        snapshot = await fill_forms_graph.aget_state(thread_config)
        if snapshot.next and "human_approval" in snapshot.next:
            interrupted_state = snapshot
            break
    
    if interrupted_state is None:
        print("\nGraph finished without requiring human approval.")
        return

    # ── Step 4: Human reviews and decides ─────────────────────────────────────
    # Retrieve the interrupt payload surfaced by human_approval_node
    interrupt_payload = None
    for task in interrupted_state.tasks:
        if task.interrupts:
            interrupt_payload = task.interrupts[0].value
            break

    if interrupt_payload:
        print("\n" + "=" * 80)
        print("HUMAN APPROVAL REQUIRED")
        print("=" * 80)
        print(interrupt_payload.get("question", ""))
        print("=" * 80)
        decision = input("\nYour decision (yes/no): ").strip()
    else:
        decision = "no"

    # ── Step 5: Resume the graph with the human decision ──────────────────────
    print(f"\nStep 5: Resuming graph with decision='{decision}'...")
    resume_events = fill_forms_graph.astream(
        Command(resume=decision),
        config=thread_config,
        stream_mode="values",
    )
    for event in resume_events:
        last_msg = event["messages"][-1]
        if hasattr(last_msg, "content") and last_msg.content:
            print(f"\n[Agent]: {last_msg.content}")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())




"""
python AgenticAI/fill_online_form/main.py

"""

