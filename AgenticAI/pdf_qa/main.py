"""
LangChain PDF QA Agent
Reads a PDF file and answers user questions using RAG (Retrieval-Augmented Generation).

Requirements:
    pip install langchain langchain-community langchain-openai langchain-text-splitters langgraph pypdf faiss-cpu openai

Usage:
    Set OPENAI_API_KEY environment variable, then run:    
    wget https://writing.colostate.edu/guides/documents/resume/functionalsample.pdf
    python main.py --pdf functionalsample.pdf


"""


import os
import argparse
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
#from langchain.agents import create_react_agent # for the new versions

######################################################################
# Module-level references set during build_agent()
_retriever = None
_llm = None

######################################################################

prompt_cover_letter = """You are an expert career coach and professional writer.

Below is extracted content from a candidate's resume:
---
{resume_context}
---

Job Role: {job_role}

Job Description:
{job_desc}

Write a compelling, personalised cover letter for this candidate applying to the above role.
The letter should:
- Open with a strong hook that mentions the role by name
- Highlight 2-3 specific experiences or achievements from the resume that match the job requirements
- Show enthusiasm for the company/role
- Close with a confident call to action
- Be concise (3-4 paragraphs), professional, and human in tone
- NOT invent any facts not present in the resume

Return only the cover letter text, ready to send."""


######################################################################

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


@tool
def write_a_cover_letter(job_role: str, job_desc: str) -> str:
    """
    Generate a professional cover letter tailored to a specific job.

    Uses the resume content from the indexed PDF to extract relevant experience,
    skills, and achievements, then writes a compelling cover letter.

    Args:
        job_role: The title of the job being applied for (e.g. 'Senior Data Scientist').
        job_desc: The full job description or a summary of requirements and responsibilities.

    Returns:
        A professionally written cover letter as a string.
    """
    if _retriever is None or _llm is None:
        raise RuntimeError("Agent not initialised — call build_agent() first.")

    # Retrieve the most relevant resume chunks for the target role
    resume_docs = _retriever.invoke(job_role + " " + job_desc)
    resume_context = "\n\n".join(doc.page_content for doc in resume_docs)

    response = _llm.invoke(prompt_cover_letter.format(resume_context=resume_context, job_role=job_role, job_desc=job_desc))
    return response.content




def build_agent(vectorstore):
    """Build a LangGraph ReAct agent with a PDF retrieval tool."""
    global _retriever, _llm
 
    _llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    _retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

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
        "You are a helpful career assistant. The user has uploaded their resume as a PDF. "
        "Use PDFSearch to answer questions about the resume. "
        "Use write_a_cover_letter when the user asks to generate a cover letter — "
        "always ask for the job role and job description first if not provided."
    )

    agent = create_react_agent(
        model=_llm,
        tools=[retriever_tool, write_a_cover_letter],
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



# for multiple pdf loads
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader

loader = DirectoryLoader(
    "path/to/pdf_folder",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

documents = loader.load()

"""