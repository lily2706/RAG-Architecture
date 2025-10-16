import os
import json
import argparse
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

PERSIST_DIR = "vectordb"
COLLECTION_NAME = "askmydocs"

SYSTEM_PROMPT = """You are a helpful assistant that answers strictly using the provided context.
If the answer is not in the context, say you don't know.
Keep answers concise and include inline citations as [Doc i].
"""

def build_chain(k=4, use_mmr=False, model="gemini-2.0-flash", temperature=0.0):
    # Embeddings + vector DB (must match ingest.py)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vectordb = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

    retriever = vectordb.as_retriever(
        search_type=("mmr" if use_mmr else "similarity"),
        search_kwargs={"k": k}
    )

    def format_docs(docs):
        """Format retrieved docs and print context before sending to LLM."""
        blocks = []
        for i, d in enumerate(docs, start=1):
            src = d.metadata.get("source", "unknown")
            page = d.metadata.get("page", None)
            tag = f"[Doc {i}: {os.path.basename(src)}"
            if isinstance(page, int):
                tag += f" p.{page+1}]"
            else:
                tag += "]"
            blocks.append(f"{tag}\n{d.page_content}")

        formatted = "\n\n---\n\n".join(blocks)

        # 🔍 Print the context so you can see what Gemini sees
        print("\n================= RETRIEVED CONTEXT =================")
        print(formatted[:3000])  # limit output to first 3 000 chars for readability
        print("=====================================================\n")

        return formatted

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer with citations.")
    ])

    # Use Gemini 2.0 Flash model (via REST transport)
    llm = ChatGoogleGenerativeAI(
        model=model,                # e.g., "gemini-2.0-flash"  ✅
        temperature=temperature,
        transport="rest",           # required to avoid gRPC 404s
    )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def run_once(question, k=4, mmr=False, model="gemini-2.0-flash", temperature=0.0, as_json=False):
    chain = build_chain(k=k, use_mmr=mmr, model=model, temperature=temperature)
    answer = chain.invoke(question)
    if as_json:
        print(json.dumps({"question": question, "answer": answer}, ensure_ascii=False, indent=2))
    else:
        print("\n=== ANSWER ===")
        print(answer)


def chat_loop(k=4, mmr=False, model="gemini-2.0-flash", temperature=0.0):
    chain = build_chain(k=k, use_mmr=mmr, model=model, temperature=temperature)
    print("Interactive mode. Type your question, or '/exit' to quit.\n")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not q:
            continue
        if q.lower() in {"/exit", "exit", "quit", ":q"}:
            print("Bye.")
            break
        ans = chain.invoke(q)
        print("\n=== ANSWER ===")
        print(ans)
        print()


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Ask your personal knowledge base using Gemini 2.0 Flash.")
    parser.add_argument("question", nargs="?", help="Your question (omit to use --chat).")
    parser.add_argument("--chat", action="store_true", help="Interactive chat loop.")
    parser.add_argument("--k", type=int, default=4, help="Top-k retrieved chunks (default 4).")
    parser.add_argument("--mmr", action="store_true", help="Use MMR retrieval (diverse results).")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash", help="Gemini model (default: gemini-2.0-flash).")
    parser.add_argument("--temp", type=float, default=0.0, help="Model temperature.")
    parser.add_argument("--json", action="store_true", help="Output JSON.")
    args = parser.parse_args()

    if not os.path.isdir(PERSIST_DIR):
        print(f"[ASK] Vector DB not found at '{PERSIST_DIR}'. Run 'python ingest.py' first.")
        return

    if args.chat:
        chat_loop(k=args.k, mmr=args.mmr, model=args.model, temperature=args.temp)
    elif args.question:
        run_once(args.question, k=args.k, mmr=args.mmr, model=args.model, temperature=args.temp, as_json=args.json)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
