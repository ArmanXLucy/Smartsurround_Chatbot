import os
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
ROOT=Path(__file__).resolve().parent.parent
knowledge=ROOT/"knowledge"; db=ROOT/"rag"/"chroma_db"
client=chromadb.PersistentClient(path=str(db)); collection=client.get_or_create_collection("smartsurround")
old=collection.get().get("ids",[])
if old: collection.delete(ids=old)
encoder=SentenceTransformer(os.getenv("EMBEDDING_MODEL","BAAI/bge-small-en-v1.5"))
docs=[]; metas=[]; ids=[]
for f in sorted(knowledge.glob("*.md")):
    for chunk in f.read_text(encoding="utf-8").split("\n\n"):
        chunk=chunk.strip()
        if len(chunk)>=30:
            docs.append(chunk); metas.append({"source":f.name}); ids.append(f"chunk_{len(ids)}")
if not docs: raise RuntimeError("No knowledge files found")
emb=encoder.encode(docs,normalize_embeddings=True).tolist()
collection.add(ids=ids,documents=docs,embeddings=emb,metadatas=metas)
print(f"Indexed {len(docs)} chunks")
