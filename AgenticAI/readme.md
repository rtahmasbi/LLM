
**There are many good examples in this directory.**

# LangGraph
https://github.com/langchain-ai/langgraph

## ChatPromptTemplate
```py
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}")
])

formatted = prompt.invoke({"question": "What is LangChain?"})
print(formatted)
# messages=[SystemMessage(content='You are a helpful assistant.', additional_kwargs={}, response_metadata={}), HumanMessage(content='What is LangChain?', additional_kwargs={}, response_metadata={})]

```

## ChatPromptTemplate and StrOutputParser
```py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI()
llm.invoke("What is the best way to learn LLM?")


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a world class technical documentation writer."),
    ("user", "{input}")
])

chain = prompt | llm 
chain.invoke({"input": "how can langsmith help with testing?"})


chain = prompt | llm | StrOutputParser()
chain.invoke({"input": "how can langsmith help with testing?"})

```

## HumanMessage, AIMessage, SystemMessage
```py
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage

model = init_chat_model("gpt-5-nano")

system_msg = SystemMessage("You are a helpful assistant.")
human_msg = HumanMessage("Hello, how are you?")

# Use with chat models
messages = [system_msg, human_msg]
response = model.invoke(messages)  # Returns AIMessage

```

## langgraph and ToolNode
```py
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# Defines agent state and manages messages
class AgentState(TypedDict):
    # Manages the sequence of messages using the add_messages reducer function
    messages: Annotated[Sequence[BaseMessage], add_messages]


workflow = StateGraph(AgentState)

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

# invoke
graph.invoke("text")

```

## Visualize the graph
```py
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))
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



## LLM with schema for structured
```py
from pydantic import BaseModel, Field

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query that is optimized web search.")
    justification: str = Field(
        None, description="Why this query is relevant to the user's request."
    )


# Augment the LLM with schema for structured output
structured_llm = llm.with_structured_output(SearchQuery)
output = structured_llm.invoke("How does Calcium CT score relate to high cholesterol?")

```

## bind_tools
```py
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Define a tool
def multiply(a: int, b: int) -> int:
    return a * b

llm_with_tools = llm.bind_tools([multiply])
msg = llm_with_tools.invoke("What is 2 times 3?")

```

## create_react_agent
check the `AgenticAI/basic_agents/test_create_react_agent.py` code.



## langchain and Ollama

```py
# pip install langchain-ollama
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.1",
    temperature=0,
    # other params...
)
print(llm("What is the capital of France?"))

```

## langchain and vllm
```py
from langchain_community.llms import VLLM

llm = VLLM(
    model="mosaicml/mpt-7b",
    trust_remote_code=True,  # mandatory for some Hugging Face models
    max_new_tokens=128,
    top_k=10,
    temperature=0.8,
    # tensor_parallel_size=... for distributed inference
)

print(llm("What is the capital of France?"))
```

## langchain and LlamaCpp
```py
#  pip install llama-cpp-python
# pip install huggingface-hub

from langchain_community.llms import LlamaCpp
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="unsloth/Llama-3.2-1B-Instruct-GGUF",
    filename="Llama-3.2-1B-Instruct-Q4_K_M.gguf",
)
#model_path="hf://bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
#model_path="/path/to/llama-3.gguf",


llm = LlamaCpp(
    model_path=model_path,
    temperature=0.7,
    max_tokens=512,
    verbose=False,
)

response = llm.invoke("What is the capital of France?")
print(response)

```



## langgraph multi agent structures
https://langchain-opentutorial.gitbook.io/langchain-opentutorial/17-langgraph/02-structures/09-langgraph-multi-agent-structures-02



https://www.marktechpost.com/2026/03/01/how-to-design-a-production-grade-multi-agent-communication-system-using-langgraph-structured-message-bus-acp-logging-and-persistent-shared-state-architecture/?amp


## PyPDFLoader and FAISS
check this `AgenticAI/pdf_qa/main.py` for `RecursiveCharacterTextSplitter`, 


## Runnable
- RunnablePassthrough
- RunnableLambda
- RunnableParallel
- RunnableBranch

### RunnableLambda
```py
from langchain_core.runnables import RunnableLambda

# A RunnableSequence constructed using the `|` operator
sequence = RunnableLambda(lambda x: x + 1) | RunnableLambda(lambda x: x * 2)
sequence.invoke(1)  # 4
sequence.batch([1, 2, 3])  # [4, 6, 8]

# A sequence that contains a RunnableParallel constructed using a dict literal
sequence = RunnableLambda(lambda x: x + 1) | {
    "mul_2": RunnableLambda(lambda x: x * 2),
    "mul_5": RunnableLambda(lambda x: x * 5),
}
sequence.invoke(1)  # {'mul_2': 4, 'mul_5': 10}
```


### RunnablePassthrough
```py
# pip install docarray
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings()

vecstore_a = DocArrayInMemorySearch.from_texts(
    ["half the info will be here", "James' birthday is the 7th December"],
    embedding=embeddings
)

retriever_a = vecstore_a.as_retriever()

retrieval = RunnableParallel(
    {"context": retriever_a, "question": RunnablePassthrough()}
)

prompt_str = """Answer the question below using the context:

Context: {context}

Question: {question}

Answer: """
prompt = ChatPromptTemplate.from_template(prompt_str)


output_parser = StrOutputParser()

rag_chain = retrieval | prompt | llm | output_parser

out = rag_chain.invoke("when was James born?")
print(out)
# James was born on the 7th of December.

```


## stream
```py
m = {
        "messages": [
            {
                "role": "user",
                "content": "What does Lilian Weng say about types of reward hacking?",
            }
        ]
}

for chunk in graph.stream(m):
    for node, update in chunk.items():
        print("Update from node", node)
        update["messages"][-1].pretty_print()
        print("\n\n")

```




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


