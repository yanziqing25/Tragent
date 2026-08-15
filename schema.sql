-- Trading-owned database schema.
-- LangGraph creates and migrates its own checkpoint/store tables at FastAPI startup.

CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(64) NOT NULL,
    password VARCHAR NOT NULL
);

CREATE UNIQUE INDEX ix_users_username ON users (username);
