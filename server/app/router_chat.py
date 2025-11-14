from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .memory import (
    init_db,
    save_message_with_embedding,
    update_conversation_summary_if_needed,
    get_recent_messages,
    get_conversation_summary,
    search_similar_messages,
    generate_reply,
)

router = APIRouter(prefix="/api/v1", tags=["chat"])


class ChatRequest(BaseModel):
    conversation_id: str
    user_text: str


class ChatResponse(BaseModel):
    reply: str
    summary: str | None = None


@router.on_event("startup")
def startup_event():
    # ensure tables exist
    init_db()


@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest):
    try:
        # 1. Save user message
        save_message_with_embedding(payload.conversation_id, "user", payload.user_text)

        # 2. Maybe update summary
        update_conversation_summary_if_needed(payload.conversation_id)

        # 3. Load memory layers
        recent = get_recent_messages(payload.conversation_id, limit=6)
        summary = get_conversation_summary(payload.conversation_id)
        semantic = search_similar_messages(payload.conversation_id, payload.user_text, limit=5)

        # 4. Generate reply
        reply_text = generate_reply(recent, payload.user_text, semantic, summary)

        # 5. Save assistant message
        save_message_with_embedding(payload.conversation_id, "assistant", reply_text)

        # 6. Return to client
        return ChatResponse(reply=reply_text, summary=summary)

    except Exception as e:
        print("⚠️ Chat endpoint error:", e)
        raise HTTPException(status_code=500, detail="Internal server error")
