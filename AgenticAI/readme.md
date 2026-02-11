
# LangGraph
https://github.com/langchain-ai/langgraph

```py
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

workflow = StateGraph(MessagesState)

# Define the nodes we will cycle between
workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)

workflow.add_edge(START, "generate_query_or_respond")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "generate_query_or_respond",
    # Assess LLM decision (call `retriever_tool` tool or respond to the user)
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

# Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
)
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

# Compile
graph = workflow.compile()

```

## Visualize the graph
```py
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))
```

## Run the agentic RAG
```py
for chunk in graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "What does Lilian Weng say about types of reward hacking?",
            }
        ]
    }
):
    for node, update in chunk.items():
        print("Update from node", node)
        update["messages"][-1].pretty_print()
        print("\n\n")

```


# claude-agent-sdk-python
https://code.claude.com/docs/en/settings#tools-available-to-claude
```
Tool	Description	Permission Required
AskUserQuestion	Asks multiple-choice questions to gather requirements or clarify ambiguity	No
Bash	Executes shell commands in your environment (see Bash tool behavior below)	Yes
TaskOutput	Retrieves output from a background task (bash shell or subagent)	No
Edit	Makes targeted edits to specific files	Yes
ExitPlanMode	Prompts the user to exit plan mode and start coding	Yes
Glob	Finds files based on pattern matching	No
Grep	Searches for patterns in file contents	No
KillShell	Kills a running background bash shell by its ID	No
MCPSearch	Searches for and loads MCP tools when tool search is enabled	No
NotebookEdit	Modifies Jupyter notebook cells	Yes
Read	Reads the contents of files	No
Skill	Executes a skill within the main conversation	Yes
Task	Runs a sub-agent to handle complex, multi-step tasks	No
TaskCreate	Creates a new task in the task list	No
TaskGet	Retrieves full details for a specific task	No
TaskList	Lists all tasks with their current status	No
TaskUpdate	Updates task status, dependencies, details, or deletes tasks	No
WebFetch	Fetches content from a specified URL	Yes
WebSearch	Performs web searches with domain filtering	Yes
Write	Creates or overwrites files	Yes
LSP	Code intelligence via language servers. Reports type errors and warnings automatically after file edits. Also supports navigation operations: jump to definitions, find references, get type info, list symbols, find implementations, trace call hierarchies. Requires a code intelligence plugin and its language server binary	No
```

