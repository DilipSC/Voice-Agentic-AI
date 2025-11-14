from sqlalchemy import create_engine, text

# Example Neon connection string:
# postgresql+psycopg2://user:password@host/dbname
DATABASE_URL = "postgresql://neondb_owner:npg_4iEelYU9aCgu@ep-weathered-hall-ah1wijmy-pooler.c-3.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

engine = create_engine(DATABASE_URL, echo=False)

def save_message(conversation_id: str, role: str, content: str):
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO messages (conversation_id, role, content)
                VALUES (:cid, :role, :content)
            """),
            {"cid": conversation_id, "role": role, "content": content},
        )

def get_recent_messages(conversation_id: str, limit: int = 10):
    with engine.begin() as conn:
        result = conn.execute(
            text("""
                SELECT role, content
                FROM messages
                WHERE conversation_id = :cid
                ORDER BY created_at DESC
                LIMIT :limit
            """),
            {"cid": conversation_id, "limit": limit},
        )
        return list(result)

if __name__ == "__main__":
    conv_id = "test_conv_1"
    save_message(conv_id, "user", "Hello, remember me?")
    save_message(conv_id, "assistant", "Yes, I remember you from last time.")
    msgs = get_recent_messages(conv_id)

    print("Recent messages (newest first):")
    for row in msgs:
        print(row.role, ":", row.content)
