from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .router_chat import router as chat_router
from .memory import init_db

load_dotenv()

app = FastAPI(title="Voice AI Backend", version="1.0.0")

# CORS for your Next.js frontend
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    print("Initializing DB...")
    init_db()
    print("DB ready ✅")


app.include_router(chat_router)


@app.get("/")
async def root():
    return {"message": "Voice AI backend running"}
