# SmartSurround Chatbot

Separate AI chatbot service for the SmartSurround project.

Architecture:
React floating chatbot -> FastAPI -> RAG (BGE + ChromaDB) -> Qwen2.5-0.5B-Instruct/Ollama.

This folder is intentionally separate from the existing SmartSurround frontend and backend.
