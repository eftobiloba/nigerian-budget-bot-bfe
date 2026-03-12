import os
import faiss
import pickle
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from pathlib import Path
from google import genai
from google.genai import types
from pydantic import BaseModel

# ---------------- CONFIG ----------------
DOCUMENTS_FOLDER = "docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
TOP_K = 3
GEMINI_MODEL = "gemini-2.5-flash-preview-05-20"
API_KEY = ""
INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "chunks.pkl"

class QuestionRequest(BaseModel):
    question: str

# Init
app = FastAPI()
embed_model = SentenceTransformer(EMBEDDING_MODEL)
client = genai.Client(api_key=API_KEY)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- HELPERS ----------------
def load_documents(folder_path):
    texts = []
    for file_path in Path(folder_path).glob("*"):
        if file_path.suffix.lower() == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                texts.append(f.read())
        elif file_path.suffix.lower() == ".pdf":
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            texts.append(text)
    return texts

def chunk_text(text):
    words = text.split()
    return [" ".join(words[i:i+CHUNK_SIZE]) for i in range(0, len(words), CHUNK_SIZE)]

def build_index():
    docs_raw = load_documents(DOCUMENTS_FOLDER)
    chunks = []
    for doc in docs_raw:
        chunks.extend(chunk_text(doc))

    embeddings = embed_model.encode(chunks, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)

def load_index():
    # If missing, build automatically
    if not Path(INDEX_FILE).exists() or not Path(CHUNKS_FILE).exists():
        print("No index found — building new index from local docs...")
        build_index()

    index = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def generate(prompt):
    client = genai.Client(
        api_key=API_KEY,
    )

    model = GEMINI_MODEL
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
    )

    # Collect all chunks from the stream
    response_text = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        response_text += chunk.text

    return response_text

# ---------------- ROUTES ----------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    save_path = Path(DOCUMENTS_FOLDER) / file.filename
    with open(save_path, "wb") as f:
        f.write(await file.read())
    build_index()
    return {"message": "File uploaded and index updated."}

@app.post("/ask")
async def ask_question(req: QuestionRequest):
    index, chunks = load_index()
    q_emb = embed_model.encode(req.question, convert_to_numpy=True)
    q_emb = q_emb.reshape(1, -1)
    _, ids = index.search(q_emb, k=TOP_K)
    retrieved_chunks = [chunks[i] for i in ids[0]]
    
    # 🩹 Fix tuple issue here:
    retrieved_chunks = [c[0] if isinstance(c, tuple) else c for c in retrieved_chunks]
    
    context = "\n\n".join(retrieved_chunks)
    prompt = f"""Answer the question based only on the following context:

{context}

Question: {req.question}
Answer:"""

    answer = generate(prompt)
    return JSONResponse({"answer": answer})


if __name__ == "__main__":
    build_index()