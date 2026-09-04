import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================
# 1. EMBEDDING MODEL
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 2. DOCUMENT INGESTION FUNCTION
# ============================================================

def build_vectorstore(pdf_path: str, output_dir: str):
    """
    Load a PDF, split it into chunks, create embeddings,
    build a FAISS vector store, and save it locally.
    """

    print(f"\nProcessing: {pdf_path}")

    # Check that PDF exists
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    # --------------------------------------------------------
    # Step 1: Load PDF
    # --------------------------------------------------------

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"Loaded {len(documents)} pages")


    # --------------------------------------------------------
    # Step 2: Split documents into chunks
    # --------------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")


    # --------------------------------------------------------
    # Step 3: Create embeddings and FAISS index
    # --------------------------------------------------------

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    print("Embeddings created and FAISS index built")


    # --------------------------------------------------------
    # Step 4: Save FAISS index
    # --------------------------------------------------------

    os.makedirs(output_dir, exist_ok=True)

    vectorstore.save_local(output_dir)

    print(f"Vector store saved to: {output_dir}")


# ============================================================
# 3. MAIN INGESTION PIPELINE
# ============================================================

if __name__ == "__main__":

    # Academic document
    build_vectorstore(
        pdf_path="data/academics_handbook.pdf",
        output_dir="vectorstores/academic"
    )

    # Fee document
    build_vectorstore(
        pdf_path="data/fee_structure.pdf",
        output_dir="vectorstores/fee"
    )

    print("\n========================================")
    print("       INGESTION COMPLETED")
    print("========================================")