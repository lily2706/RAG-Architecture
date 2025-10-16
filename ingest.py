import os
from dotenv import load_dotenv

# Loaders for various file types
from langchain_community.document_loaders import (
    PyPDFDirectoryLoader, DirectoryLoader, TextLoader, BSHTMLLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Gemini embeddings + Chroma vector DB
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# === CONFIG ===
DATA_DIR = "data"                # folder containing your PDFs / notes / docs
PERSIST_DIR = "vectordb"         # folder where the vector index is stored
COLLECTION_NAME = "askmydocs"    # name of the Chroma collection
CHUNK_SIZE = 6000                # large chunk size so resumes stay intact
CHUNK_OVERLAP = 0                # no overlap since we use large chunks


def load_docs():
    """Load all supported document types from DATA_DIR."""
    docs = []
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"[INGEST] Created folder '{DATA_DIR}'. Add your PDFs or TXT files there.")
        return docs

    # PDFs
    if any(f.lower().endswith(".pdf") for f in os.listdir(DATA_DIR)):
        print("[INGEST] Loading PDF files...")
        docs += PyPDFDirectoryLoader(DATA_DIR).load()

    # Text / Markdown
    docs += DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader, show_progress=True).load()
    docs += DirectoryLoader(DATA_DIR, glob="**/*.md", loader_cls=TextLoader, show_progress=True).load()

    # HTML / HTM
    docs += DirectoryLoader(DATA_DIR, glob="**/*.html", loader_cls=BSHTMLLoader, show_progress=True).load()
    docs += DirectoryLoader(DATA_DIR, glob="**/*.htm",  loader_cls=BSHTMLLoader, show_progress=True).load()

    return docs


def main():
    load_dotenv()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    docs = load_docs()
    if not docs:
        print(f"[INGEST] No documents found in '{DATA_DIR}'. Please add files and re-run.")
        return

    # Summary of loaded docs
    by_ext = {}
    for d in docs:
        src = (d.metadata.get("source") or "").lower()
        ext = os.path.splitext(src)[1]
        by_ext[ext] = by_ext.get(ext, 0) + 1
    print("[INGEST] Loaded docs by type:", by_ext)

    # === Text splitting (large chunks to preserve full resume context) ===
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    print(f"[INGEST] Split into {len(chunks)} chunks (chunk size: {CHUNK_SIZE}, overlap: {CHUNK_OVERLAP})")

    # Debug preview of first chunk
    if chunks:
        print("\n--- SAMPLE CHUNK PREVIEW ---")
        print(chunks[0].page_content[:800])
        print("\n-----------------------------\n")

    # === Create / update Chroma vector store ===
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
    )

    print(f"[INGEST] ✅ Indexed {len(chunks)} chunks into '{PERSIST_DIR}' (collection: '{COLLECTION_NAME}')")
    print("[INGEST] Ready for querying with ask.py")


if __name__ == "__main__":
    main()
