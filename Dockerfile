# One image for every Python process: the API, the Celery worker and beat, the broker, and the
# migrations. Compose gives each its own command.
# Pinned to 3.14 until confluent-kafka and zstandard publish wheels for Python 3.15; then
# python:slim needs no compiler.
FROM python:3.14-slim

WORKDIR /app
COPY pyproject.toml README.md alembic.ini ./
COPY migrations ./migrations
COPY src ./src
RUN pip install --no-cache-dir .
