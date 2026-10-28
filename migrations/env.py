"""Apply migrations to the PostgreSQL database configured in .env."""

from alembic import context
from sqlalchemy import URL, create_engine

from tragent.config import load_settings

settings = load_settings()
url = URL.create(
    "postgresql+psycopg",
    username=settings.postgres_user,
    password=settings.postgres_password.get_secret_value(),
    host=settings.postgres_host,
    port=settings.postgres_port,
    database=settings.postgres_database,
)
with create_engine(url).connect() as connection:
    context.configure(connection=connection)
    with context.begin_transaction():
        context.run_migrations()
