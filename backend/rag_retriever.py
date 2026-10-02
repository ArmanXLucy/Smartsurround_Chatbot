import os
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()
ROOT=Path(__file__).resolve().parent.parent
client=chromadb.PersistentClient(path=os.getenv("CHROMA_PATH",str(ROOT/"rag"/"chroma_db")))
collection=client.get_or_create_collection("smartsurround")
encoder=SentenceTransformer(os.getenv("EMBEDDING_MODEL","BAAI/bge-small-en-v1.5"))
def retrieve_context(query):
    if collection.count()==0: return "No SmartSurround knowledge indexed yet.",[]
    emb=encoder.encode([query],normalize_embeddings=True).tolist()
    result=collection.query(query_embeddings=emb,n_results=int(os.getenv("TOP_K","4")))
    docs=result.get("documents",[[]])[0]; metas=result.get("metadatas",[[]])[0]
    return "\n\n---\n\n".join(docs),sorted(set(m.get("source") for m in metas if m and m.get("source")))
