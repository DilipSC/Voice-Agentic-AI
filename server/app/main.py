from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .router_chat import router as chat_router

from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="Voice AI Assistant API")

# Allow your Next.js frontend origin
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    # add your prod domain here later
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(chat_router)
