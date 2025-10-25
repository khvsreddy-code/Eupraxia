"""Prepare knowledge: read text files from ./data, chunk them, compute embeddings and save to Chroma
Run from backend/rag_server/ after activating venv
"""
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

DATA_DIR = Path('./data')
CHROMA_DIR = './chroma_db'
EMBED_MODEL = 'all-MiniLM-L6-v2'

print('Loading embedding model...')
model = SentenceTransformer(EMBED_MODEL)
client = chromadb.Client(Settings(chroma_db_impl='duckdb+parquet', persist_directory=CHROMA_DIR))
collection = client.get_or_create_collection('documents')

# Simple chunker
def chunk_text(text, chunk_size_words=300, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = ' '.join(words[i:i+chunk_size_words])
        chunks.append(chunk)
        i += chunk_size_words - overlap
    return chunks

count = 0
for p in DATA_DIR.glob('**/*.txt'):
    print('Processing', p)
    text = p.read_text(encoding='utf-8')
    chunks = chunk_text(text)
    ids = [f"{p.stem}_{i}" for i in range(len(chunks))]
    embs = model.encode(chunks, show_progress_bar=True)
    embs_list = [e.tolist() for e in embs]
    metas = [{'source': str(p), 'chunk_index': i} for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks, metadatas=metas, embeddings=embs_list)
    count += len(chunks)

client.persist()
print('Done. Ingested', count, 'chunks')
