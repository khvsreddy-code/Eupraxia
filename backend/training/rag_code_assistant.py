"""
RAG Code Assistant: Outperform Copilot with your own codebase context
"""
import os
import glob
from chromadb import Client
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import openai

# 1. Indexing: Embed your codebase
CODE_DIR = "../../"  # Adjust path to your repo root
CODE_EXTENSIONS = [".py", ".js", ".ts", ".jsx", ".md"]

# Load all code files
def load_code_files(code_dir, extensions):
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(code_dir, f"**/*{ext}"), recursive=True))
    return files

# Chunk code files (simple line-based)
def chunk_code(file_path, chunk_size=20):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    return ["".join(lines[i:i+chunk_size]) for i in range(0, len(lines), chunk_size)]

# 2. Embedding and Vector DB setup
client = Client(Settings(anonymized_telemetry=False))
collection = client.create_collection("codebase")
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Fast, small embedding model

print("Indexing codebase...")
code_files = load_code_files(CODE_DIR, CODE_EXTENSIONS)
chunk_id = 0
for file_path in code_files:
    chunks = chunk_code(file_path)
    for chunk in chunks:
        emb = embedder.encode(chunk)
        collection.add(
            documents=[chunk],
            embeddings=[emb],
            ids=[str(chunk_id)]
        )
        chunk_id += 1
print(f"Indexed {chunk_id} code chunks.")

# 3. Retrieval

def retrieve_context(query, top_k=5):
    query_emb = embedder.encode(query)
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k
    )
    return results["documents"][0]

# 4. Prompt Engineering

def build_prompt(context_chunks, user_query):
    context = "\n\n".join(context_chunks)
    prompt = f"""
You are an expert developer. Use only the provided context for code generation.

### CONTEXT FROM INTERNAL REPOSITORIES:
{context}
### USER REQUEST:
{user_query}
"""
    return prompt

# 5. Generation (OpenAI API example)
openai.api_key = os.getenv("OPENAI_API_KEY")  # Set your key in environment

def generate_code(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Or "gpt-4" if available
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.2
    )
    return response["choices"][0]["message"]["content"]

# Example usage
if __name__ == "__main__":
    user_query = "Write a function to connect to our internal logging service."
    context_chunks = retrieve_context(user_query, top_k=5)
    prompt = build_prompt(context_chunks, user_query)
    print("\n--- FINAL PROMPT ---\n")
    print(prompt)
    print("\n--- GENERATED CODE ---\n")
    code = generate_code(prompt)
    print(code)
