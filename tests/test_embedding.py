from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
import numpy as np

DATABASE_URL = "postgresql://neondb_owner:npg_4iEelYU9aCgu@ep-weathered-hall-ah1wijmy-pooler.c-3.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
engine = create_engine(DATABASE_URL, echo=False)
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(texts):
    return model.encode(texts, normalize_embeddings=True)

def add_message_with_embedding(conversation_id, role, content):
    emb = embed([content])[0].tolist()
    with engine.begin() as conn:
        # 1) insert into messages, returning id
        msg_id = conn.execute(
            text("""
                INSERT INTO messages (conversation_id, role, content)
                VALUES (:cid, :role, :content)
                RETURNING id
            """),
            {"cid": conversation_id, "role": role, "content": content},
        ).scalar_one()

        # 2) insert embedding
        conn.execute(
            text("""
                INSERT INTO message_embeddings (message_id, embedding)
                VALUES (:mid, :embedding)
            """),
            {"mid": msg_id, "embedding": emb},
        )

def search_similar(query, limit=5):
    q_emb = embed([query])[0].tolist()
    with engine.begin() as conn:
        result = conn.execute(
            text("""
                SELECT m.content, 1 - (me.embedding <=> :q_emb) AS similarity
                FROM message_embeddings me
                JOIN messages m ON me.message_id = m.id
                ORDER BY me.embedding <=> :q_emb
                LIMIT :limit
            """),
            {"q_emb": q_emb, "limit": limit},
        )
        return list(result)

if __name__ == "__main__":
    conv_id = "test_conv_embed"
    add_message_with_embedding(conv_id, "user", "My favorite programming language is Python.")
    add_message_with_embedding(conv_id, "user", "I live in Bangalore.")
    add_message_with_embedding(conv_id, "assistant", "Nice! I will remember that.")

    hits = search_similar("Where do I live?")
    for row in hits:
        print(f"{row.similarity:.3f} -> {row.content}")
