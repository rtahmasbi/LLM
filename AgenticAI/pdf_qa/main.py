"""
LangChain PDF QA Agent
Reads a PDF file and answers user questions using RAG (Retrieval-Augmented Generation).

Requirements:
    pip install langchain langchain-community langchain-openai pypdf faiss-cpu openai
    
Usage:
    Set OPENAI_API_KEY environment variable, then run:    
    python main.py --pdf path/to/your/file.pdf

"""


import os
import argparse
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent


def load_and_index_pdf(pdf_path: str):
    """Load PDF with PyPDF and index it into a FAISS vector store."""
    print(f"📄 Loading PDF: {pdf_path}")

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


def build_agent(vectorstore):
    """Build a LangGraph ReAct agent with a PDF retrieval tool."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Create a retriever tool from the vector store
    retriever_tool = create_retriever_tool(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        name="PDFSearch",
        description=(
            "Use this tool to search and answer questions about the content of the PDF document. "
            "Input should be a clear, concise question or keyword."
        ),
    )

    system_prompt = (
        "You are a helpful assistant that answers questions based on a PDF document. "
        "Always use the PDFSearch tool to find relevant content before answering."
    )

    # create_react_agent from langgraph handles tool-calling natively
    agent = create_react_agent(
        model=llm,
        tools=[retriever_tool],
        prompt=system_prompt,
    )
    return agent


def interactive_loop(agent):
    """Run an interactive Q&A loop with conversation history."""
    print("=" * 60)
    print("PDF Q&A Agent ready. Type 'exit' or 'quit' to stop.")
    print("=" * 60)

    chat_history = []

    while True:
        try:
            question = input("\n🙋 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        chat_history.append(HumanMessage(content=question))

        result = agent.invoke({"messages": chat_history})
        response = result["messages"][-1].content

        chat_history.append(AIMessage(content=response))
        print(f"\n🤖 Agent: {response}")


def main():
    parser = argparse.ArgumentParser(description="LangChain PDF QA Agent")
    parser.add_argument(
        "--pdf",
        type=str,
        required=True,
        help="Path to the PDF file to query",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.pdf):
        raise FileNotFoundError(f"PDF not found: {args.pdf}")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set.\n"
            "Export it with:  export OPENAI_API_KEY='sk-...'"
        )

    vectorstore = load_and_index_pdf(args.pdf)
    agent = build_agent(vectorstore)
    interactive_loop(agent)


if __name__ == "__main__":
    main()



"""

python AgenticAI/pdf_qa/main.py --pdf /home/ras/functionalsample.pdf


"""