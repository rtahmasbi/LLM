
**There are many good examples in this directory.**


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


## deepagent langchain
https://docs.langchain.com/oss/python/deepagents/overview



## middleware langchain
```py
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
```
Middleware provides a way to more tightly control what happens inside the agent. Middleware is useful for the following:
- Tracking agent behavior with **logging**, analytics, and debugging.
- Transforming prompts, tool selection, and output formatting.
- Adding retries, fallbacks, and early termination logic.
- Applying rate limits, guardrails, and **PII detection**.


```py
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, HumanInTheLoopMiddleware

agent = create_agent(
    model="gpt-4.1",
    tools=[...],
    middleware=[
        SummarizationMiddleware(...),
        HumanInTheLoopMiddleware(...)
    ],
)
````


| Middleware           | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| Summarization        | Automatically summarize conversation history when approaching token limits. |
| Human-in-the-loop    | Pause execution for human approval of tool calls.                           |
| Model call limit     | Limit the number of model calls to prevent excessive costs.                 |
| Tool call limit      | Control tool execution by limiting call counts.                             |
| Model fallback       | Automatically fallback to alternative models when primary fails.            |
| PII detection        | Detect and handle Personally Identifiable Information (PII).                |
| To-do list           | Equip agents with task planning and tracking capabilities.                  |
| LLM tool selector    | Use an LLM to select relevant tools before calling main model.              |
| Tool retry           | Automatically retry failed tool calls with exponential backoff.             |
| Model retry          | Automatically retry failed model calls with exponential backoff.            |
| LLM tool emulator    | Emulate tool execution using an LLM for testing purposes.                   |
| Context editing      | Manage conversation context by trimming or clearing tool uses.              |
| Shell tool           | Expose a persistent shell session to agents for command execution.          |
| File search          | Provide Glob and Grep search tools over filesystem files.                   |
| Filesystem           | Provide agents with a filesystem for storing context and long-term memories.|
| Subagent             | Add the ability to spawn subagents.                                         |



# claude
## claude-agent-sdk-python
https://code.claude.com/docs/en/settings#tools-available-to-claude

| Tool            | Description | Permission Required |
|-----------------|------------|---------------------|
| AskUserQuestion | Asks multiple-choice questions to gather requirements or clarify ambiguity | No |
| Bash | Executes shell commands in your environment (see Bash tool behavior below) | Yes |
| TaskOutput | Retrieves output from a background task (bash shell or subagent) | No |
| Edit | Makes targeted edits to specific files | Yes |
| ExitPlanMode | Prompts the user to exit plan mode and start coding | Yes |
| Glob | Finds files based on pattern matching | No |
| Grep | Searches for patterns in file contents | No |
| KillShell | Kills a running background bash shell by its ID | No |
| MCPSearch | Searches for and loads MCP tools when tool search is enabled | No |
| NotebookEdit | Modifies Jupyter notebook cells | Yes |
| Read | Reads the contents of files | No |
| Skill | Executes a skill within the main conversation | Yes |
| Task | Runs a sub-agent to handle complex, multi-step tasks | No |
| TaskCreate | Creates a new task in the task list | No |
| TaskGet | Retrieves full details for a specific task | No |
| TaskList | Lists all tasks with their current status | No |
| TaskUpdate | Updates task status, dependencies, details, or deletes tasks | No |
| WebFetch | Fetches content from a specified URL | Yes |
| WebSearch | Performs web searches with domain filtering | Yes |
| Write | Creates or overwrites files | Yes |
| LSP | Code intelligence via language servers. Reports type errors and warnings automatically after file edits. Also supports navigation operations: jump to definitions, find references, get type info, list symbols, find implementations, trace call hierarchies. Requires a code intelligence plugin and its language server binary | No |

