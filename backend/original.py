import os
import faiss
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from pathlib import Path
import textwrap

# ----------------------
# CONFIGURATION
# ----------------------
DOCUMENTS_FOLDER = "docs"  # Folder containing your PDFs and/or TXT files
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500  # Approx tokens per chunk
TOP_K = 3         # How many chunks to send to Gemini
GEMINI_MODEL = "gemini-2.5-flash-preview-05-20"
API_KEY = "AIzaSyCxqn3gMIfrR1Ya8hGuZyt4HaqsQNbLyvA"

# ----------------------
# INITIALIZE
# ----------------------
embed_model = SentenceTransformer(EMBEDDING_MODEL)

# ----------------------
# HELPER FUNCTIONS
# ----------------------
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

def chunk_text(text, chunk_size=CHUNK_SIZE):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

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

# ----------------------
# LOAD & PROCESS DOCS
# ----------------------
print("Loading documents...")
docs_raw = load_documents(DOCUMENTS_FOLDER)

print("Chunking documents...")
chunks = []
for doc in docs_raw:
    chunks.extend(chunk_text(doc))

print(f"Total chunks: {len(chunks)}")

# ----------------------
# EMBEDDINGS & INDEX
# ----------------------
print("Encoding chunks...")
embeddings = embed_model.encode(chunks, convert_to_numpy=True)

print("Building FAISS index...")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# ----------------------
# QUERY LOOP
# ----------------------
while True:
    query = input("\nAsk a question (or type 'exit'): ")
    if query.lower() == "exit":
        break

    # 1. Encode query
    q_emb = embed_model.encode([query], convert_to_numpy=True)

    # 2. Retrieve top chunks
    _, ids = index.search(q_emb, k=TOP_K)
    retrieved_chunks = [chunks[i] for i in ids[0]]

    # 3. Build prompt for Gemini
    context = "\n\n".join(retrieved_chunks)
    prompt = f"""Answer the question based only on the following context:

{context}

Question: {query}
Answer:"""

    response = generate(prompt=prompt)

    # 5. Output
    print("\n" + "-"*50)
    print(textwrap.fill(response, width=80))
    print("-"*50)