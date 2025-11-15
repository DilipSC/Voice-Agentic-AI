from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .memory import (
    save_message_with_embedding,
    get_recent_messages,
    get_conversation_summary,
    search_similar_messages,
    update_conversation_summary_if_needed,
)
from .agent.agent_brain import run_agentic_response

router = APIRouter(prefix="/api/v1", tags=["chat"])


class ChatRequest(BaseModel):
    conversation_id: str
    user_text: str


@router.post("/chat")
async def chat_endpoint(body: ChatRequest):
    cid = body.conversation_id
    user_text = body.user_text

    try:
        # 1) Save user message + embedding
        save_message_with_embedding(cid, "user", user_text)

        # 2) Maybe update long-term summary
        update_conversation_summary_if_needed(cid)

        # 3) Fetch memory layers
        recent = get_recent_messages(cid, limit=6)
        summary = get_conversation_summary(cid)
        semantic = search_similar_messages(cid, user_text, limit=5)

        # 4) Build context text for agent
        context_lines = []
        if summary:
            context_lines.append(f"Long-term summary:\n{summary}")

        if recent:
            ctx_recent = "Recent conversation:\n"
            for m in recent:
                ctx_recent += f"{m.role.upper()}: {m.content}\n"
            context_lines.append(ctx_recent)

        if semantic:
            ctx_sem = "Relevant past memories:\n"
            seen = set()
            for row in semantic:
                c = row.content if hasattr(row, "content") else row["content"]
                if c in seen:
                    continue
                seen.add(c)
                ctx_sem += f"- {c}\n"
            context_lines.append(ctx_sem)

        context_text = "\n\n".join(context_lines) if context_lines else None

        # 5) Agentic LLM call
        reply = run_agentic_response(user_text, context_text=context_text)

        # 6) Save assistant message
        save_message_with_embedding(cid, "assistant", reply)

        return {"reply": reply}

    except Exception as e:
        print("⚠️ Chat endpoint error:", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health():
    return {"status": "ok"}
