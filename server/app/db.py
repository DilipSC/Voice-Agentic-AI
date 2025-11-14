from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from .config import get_settings

settings = get_settings()

engine: Engine = create_engine(
    settings.database_url,
    echo=False,
    pool_pre_ping=True,
)
