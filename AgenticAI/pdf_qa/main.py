"""
LangChain PDF QA Agent
Reads a PDF file and answers user questions using RAG (Retrieval-Augmented Generation).

Requirements:
    pip install langchain langchain-community langchain-openai pypdf faiss-cpu openai
    
Usage:
    Set OPENAI_API_KEY environment variable, then run:
    python pdf_agent.py --pdf path/to/your/file.pdf
"""

import os
import argparse
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory


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
    """Build a LangChain agent with a PDF retrieval tool."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # RetrievalQA chain wraps the vector store for question answering
    retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=False,
    )

    # Expose the chain as a Tool so the agent can decide when to use it
    pdf_tool = Tool(
        name="PDFSearch",
        func=retrieval_chain.run,
        description=(
            "Use this tool to answer questions about the content of the PDF document. "
            "Input should be a clear, concise question."
        ),
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    agent = initialize_agent(
        tools=[pdf_tool],
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
    )
    return agent


def interactive_loop(agent):
    """Run an interactive Q&A loop."""
    print("=" * 60)
    print("PDF Q&A Agent ready. Type 'exit' or 'quit' to stop.")
    print("=" * 60)

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

        response = agent.run(question)
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
python main.py --pdf /home/ras/functionalsample.pdf


"""