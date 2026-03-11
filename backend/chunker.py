import os
import pickle
import faiss
import numpy as np
from pathlib import Path
from pdf2image import convert_from_path
import pytesseract
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from joblib import Parallel, delayed
from typing import List, Tuple
import logging
import fitz  # PyMuPDF

# ---------------- CONFIG ----------------
DOCUMENTS_FOLDER = "docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "chunks.pkl"
BATCH_SIZE = 32
N_JOBS = -1  # Use all cores

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------- GLOBAL (Lazy) MODEL ----------------
_model = None  # Each worker process will initialize its own model


def get_model() -> SentenceTransformer:
    """Lazy load the SentenceTransformer model inside each worker."""
    global _model
    if _model is None:
        logger.info("Loading SentenceTransformer model in worker...")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


# ---------------- HELPERS ----------------
def extract_text_with_ocr(file_path: Path) -> str:
    """Use OCR (Tesseract) to extract text from a scanned or image-only PDF."""
    try:
        logger.info(f"Running OCR on {file_path.name}...")
        images = convert_from_path(file_path)
        text = ""
        for i, img in enumerate(images):
            ocr_text = pytesseract.image_to_string(img)
            if ocr_text.strip():
                text += ocr_text + "\n"
            else:
                logger.warning(f"OCR returned no text on page {i+1} of {file_path.name}")
        return text
    except Exception as e:
        logger.error(f"OCR failed for {file_path.name}: {e}")
        return ""


def extract_text_from_pdf(file_path: Path) -> str:
    """Extract text from a PDF using PyMuPDF (fitz), fallback to OCR if needed."""
    try:
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                page_text = page.get_text("text")
                if page_text:
                    text += page_text + "\n"

        if text.strip():
            return text  # ✅ regular PDF with text layer

        # 🧩 fallback to OCR
        logger.warning(f"No extractable text found in {file_path.name}, trying OCR...")
        ocr_text = extract_text_with_ocr(file_path)
        if ocr_text.strip():
            logger.info(f"OCR succeeded for {file_path.name}")
            return ocr_text
        else:
            logger.warning(f"OCR failed or returned empty text for {file_path.name}")
            return ""
    except Exception as e:
        logger.error(f"Failed to extract text from {file_path.name}: {e}")
        return ""


def load_documents(folder_path: str) -> List[Tuple[str, str]]:
    """Load text and PDF documents from the specified folder."""
    documents = []
    folder = Path(folder_path)
    for file_path in folder.glob("*"):
        try:
            if file_path.suffix.lower() == ".txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    documents.append((file_path.name, f.read()))

            elif file_path.suffix.lower() == ".pdf":
                text = extract_text_from_pdf(file_path)
                if text.strip():
                    documents.append((file_path.name, text))
                else:
                    logger.warning(f"No text extracted from {file_path.name}, skipping.")
        except Exception as e:
            logger.warning(f"Skipping unreadable file {file_path.name}: {e}")
            continue

    logger.info(f"Loaded {len(documents)} valid documents from '{folder_path}'")
    return documents


def chunk_document(doc: Tuple[str, str]) -> List[Tuple[str, str]]:
    """Chunk a single document into smaller pieces with overlap."""
    filename, text = doc
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_text(text)
    logger.info(f"Chunked {filename} into {len(chunks)} chunks")
    return [(filename, chunk) for chunk in chunks]


def embed_chunks(chunks: List[str]) -> np.ndarray:
    """Embed a list of chunks using the local worker model."""
    model = get_model()
    embeddings = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        try:
            batch_embeddings = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            embeddings.append(batch_embeddings)
        except Exception as e:
            logger.error(f"Error embedding batch {i // BATCH_SIZE}: {e}")
    return np.vstack(embeddings) if embeddings else np.array([])


def process_document(doc: Tuple[str, str]) -> Tuple[List[Tuple[str, str]], np.ndarray]:
    """Process a single document: chunk it and embed the chunks."""
    chunks = chunk_document(doc)
    chunk_texts = [chunk[1] for chunk in chunks]
    embeddings = embed_chunks(chunk_texts)
    return chunks, embeddings


def build_index():
    """Build and save a FAISS index from all documents in the folder."""
    logger.info("Starting index building process...")

    # Load documents
    documents = load_documents(DOCUMENTS_FOLDER)
    if not documents:
        logger.error("No documents found or readable. Exiting.")
        return

    # Process documents in parallel (no model passed)
    logger.info("Processing documents in parallel...")
    results = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(process_document)(doc) for doc in documents
    )

    # Combine results
    all_chunks = []
    all_embeddings = []
    for chunks, embeddings in results:
        if embeddings.size == 0:
            logger.warning("Skipping document with no embeddings.")
            continue
        all_chunks.extend(chunks)
        all_embeddings.append(embeddings)

    if not all_chunks:
        logger.error("No chunks created. Exiting.")
        return

    # Stack embeddings
    all_embeddings = np.vstack(all_embeddings)
    logger.info(f"Created {len(all_chunks)} chunks with {all_embeddings.shape[0]} embeddings")

    # Build FAISS index
    dimension = all_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(all_embeddings)
    logger.info("FAISS index built successfully")

    # Save index and chunks
    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(all_chunks, f)
    logger.info(f"Saved index to {INDEX_FILE} and chunks to {CHUNKS_FILE}")


if __name__ == "__main__":
    os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)
    build_index()
