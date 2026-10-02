import os
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_retriever import retrieve_context

load_dotenv()
app = FastAPI(title="SmartSurround AI API")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173","http://127.0.0.1:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
OLLAMA_BASE_URL=os.getenv("OLLAMA_BASE_URL","http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL=os.getenv("OLLAMA_MODEL","qwen2.5:0.5b")
class ChatRequest(BaseModel):
    message: str
    conversation_id: str|None=None
@app.get("/health")
def health(): return {"status":"ok","model":OLLAMA_MODEL}
@app.post("/chat")
def chat(req: ChatRequest):
    if not req.message.strip(): raise HTTPException(400,"Message cannot be empty")
    context,sources=retrieve_context(req.message)
    prompt=f"""You are SmartSurround AI. Answer clearly and concisely. Use the supplied SmartSurround knowledge. Do not invent SmartSurround-specific facts. If the knowledge is insufficient, say so.\n\nKNOWLEDGE:\n{context}\n\nUSER:\n{req.message}"""
    try:
        r=requests.post(f"{OLLAMA_BASE_URL}/api/generate",json={"model":OLLAMA_MODEL,"prompt":prompt,"stream":False,"options":{"temperature":0.3}},timeout=120)
        r.raise_for_status()
        data=r.json()
    except requests.RequestException as e:
        raise HTTPException(503,f"Model server unavailable: {e}")
    return {"answer":data.get("response","").strip(),"conversation_id":req.conversation_id,"sources":sources}
