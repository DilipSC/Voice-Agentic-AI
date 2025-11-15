import time
from typing import Optional, List

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

from .db import engine
from .config import get_settings

settings = get_settings()

# ---------------- Gemini for summaries ----------------
genai.configure(api_key=settings.gemini_api_key)
summary_model = genai.GenerativeModel("models/gemini-2.0-flash-001")

# ---------------- Embeddings ----------------
embedder = SentenceTransformer(settings.embedding_model_name)


def init_db():
    """Create tables if not exist. Assumes pgvector extension is enabled."""
    with engine.begin() as conn:
        conn.execute(text("""
          CREATE TABLE IF NOT EXISTS messages (
              id BIGSERIAL PRIMARY KEY,
              conversation_id VARCHAR(64),
              role VARCHAR(16),
              content TEXT,
              created_at TIMESTAMPTZ DEFAULT NOW()
          );
        """))

        conn.execute(text("""
          CREATE TABLE IF NOT EXISTS conversation_state (
              conversation_id VARCHAR(64) PRIMARY KEY,
              summary TEXT,
              last_summarized_message_id BIGINT
          );
        """))

        conn.execute(text("""
          CREATE TABLE IF NOT EXISTS message_embeddings (
              id BIGSERIAL PRIMARY KEY,
              message_id BIGINT REFERENCES messages(id) ON DELETE CASCADE,
              embedding vector(384)
          );
        """))


def get_embedding(text_value: str) -> List[float]:
    emb = embedder.encode([text_value], normalize_embeddings=True)[0]
    return emb.tolist()


def to_pgvector(emb_list: List[float]) -> str:
    # pgvector literal format: [0.1,0.2,...]
    return "[" + ",".join(str(x) for x in emb_list) + "]"


def save_message_with_embedding(
    conversation_id: str,
    role: str,
    content: str,
    engine_: Engine = engine,
) -> int:
    emb = get_embedding(content)

    with engine_.begin() as conn:
        msg_id = conn.execute(
            text("""
              INSERT INTO messages (conversation_id, role, content)
              VALUES (:cid, :role, :content)
              RETURNING id;
            """),
            {"cid": conversation_id, "role": role, "content": content},
        ).scalar_one()

        conn.execute(
            text("""
              INSERT INTO message_embeddings (message_id, embedding)
              VALUES (:mid, :embedding);
            """),
            {"mid": msg_id, "embedding": to_pgvector(emb)},
        )

    return msg_id


def get_recent_messages(conversation_id: str, limit: int = 6):
    with engine.begin() as conn:
        rows = conn.execute(
            text("""
              SELECT id, role, content
              FROM messages
              WHERE conversation_id = :cid
              ORDER BY created_at DESC
              LIMIT :limit;
            """),
            {"cid": conversation_id, "limit": limit},
        ).fetchall()

    return list(rows)[::-1]  # oldest -> newest


def get_conversation_summary(conversation_id: str) -> Optional[str]:
    with engine.begin() as conn:
        return conn.execute(
            text("""
              SELECT summary
              FROM conversation_state
              WHERE conversation_id = :cid;
            """),
            {"cid": conversation_id},
        ).scalar_one_or_none()


def search_similar_messages(conversation_id: str, query_text: str, limit: int = 5):
    q_emb = get_embedding(query_text)

    with engine.begin() as conn:
        rows = conn.execute(
            text("""
              SELECT m.id, m.role, m.content,
                     1 - (me.embedding <=> :q_emb) AS similarity
              FROM message_embeddings me
              JOIN messages m ON m.id = me.message_id
              WHERE m.conversation_id = :cid
              ORDER BY me.embedding <=> :q_emb
              LIMIT :limit;
            """),
            {
                "cid": conversation_id,
                "q_emb": to_pgvector(q_emb),
                "limit": limit,
            },
        ).fetchall()

    return list(rows)


def update_conversation_summary_if_needed(
    conversation_id: str,
    min_new_messages: int = 6,
):
    with engine.begin() as conn:
        state = conn.execute(
            text("""
              SELECT summary, last_summarized_message_id
              FROM conversation_state
              WHERE conversation_id = :cid;
            """),
            {"cid": conversation_id},
        ).mappings().first()

    existing_summary = state["summary"] if state else None
    last_id = state["last_summarized_message_id"] if state else None

    with engine.begin() as conn:
        if last_id is None:
            new_msgs = conn.execute(
                text("""
                  SELECT id, role, content
                  FROM messages
                  WHERE conversation_id = :cid
                  ORDER BY id ASC;
                """),
                {"cid": conversation_id},
            ).mappings().all()
        else:
            new_msgs = conn.execute(
                text("""
                  SELECT id, role, content
                  FROM messages
                  WHERE conversation_id = :cid
                  AND id > :last_id
                  ORDER BY id ASC;
                """),
                {"cid": conversation_id, "last_id": last_id},
            ).mappings().all()

    if len(new_msgs) < min_new_messages:
        return

    msgs_block = "".join(f"{m['role']}: {m['content']}\n" for m in new_msgs)

    prompt = f"""
You are maintaining a long-term memory summary for a user–assistant conversation.

Existing summary (may be empty):
{existing_summary or "None yet."}

New messages to incorporate:
{msgs_block}

Update the summary, keep it under 200 words. Keep:
- User preferences, habits, goals
- Important context for future turns
- Ongoing tasks or projects

Return ONLY the summary text.
"""

    try:
        time.sleep(0.3)
        updated_summary = summary_model.generate_content(prompt).text.strip()
    except Exception as e:
        print("⚠️ Summary update failed:", e)
        return

    last_msg_id = new_msgs[-1]["id"]

    with engine.begin() as conn:
        conn.execute(
            text("""
              INSERT INTO conversation_state (conversation_id, summary, last_summarized_message_id)
              VALUES (:cid, :summary, :last_id)
              ON CONFLICT (conversation_id)
              DO UPDATE SET
                summary = EXCLUDED.summary,
                last_summarized_message_id = EXCLUDED.last_summarized_message_id;
            """),
            {"cid": conversation_id, "summary": updated_summary, "last_id": last_msg_id},
        )
