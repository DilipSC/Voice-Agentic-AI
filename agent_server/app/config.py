import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
  def __init__(self):
    self.gemini_api_key = 'AIzaSyC3vdtHHX3ltPFtE3cN_bbqFfLoP0QX2nk'
    self.database_url = 'postgresql://postgres:dilip$004@db.xcsnwsgbmzowefxzljij.supabase.co:5432/postgres'
    self.embedding_model_name = "all-MiniLM-L6-v2"
    self.tavily_api_key = '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'


_settings = Settings()


def get_settings() -> Settings:
  return _settings
