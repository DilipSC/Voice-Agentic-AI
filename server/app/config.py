import os
from pydantic import BaseModel


class Settings(BaseModel):
    database_url: str
    gemini_api_key: str
    embedding_model_name: str = "all-MiniLM-L6-v2"

    class Config:
        arbitrary_types_allowed = True


def get_settings() -> Settings:
    return Settings(
        database_url='postgresql://postgres:dilip$004@db.xcsnwsgbmzowefxzljij.supabase.co:5432/postgres',
        gemini_api_key='AIzaSyC3vdtHHX3ltPFtE3cN_bbqFfLoP0QX2nk',
        embedding_model_name="all-MiniLM-L6-v2",
    )
