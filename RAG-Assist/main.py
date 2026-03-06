import uvicorn
import asyncio
from fastapi import FastAPI
from src.routes import router

app = FastAPI(
    title="rag_chatbot",
    description="A RAG-based chatbot, utilizing Groq and Gemini LLMs for comprehensive legal analysis and responses.",
    version="1.0.0"
)
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)