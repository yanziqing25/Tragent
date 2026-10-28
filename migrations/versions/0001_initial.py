"""Users, their broker accounts, orders, deployed strategies, and imported statements."""

from alembic import op

revision = "0001"
down_revision = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(64) NOT NULL,
            password VARCHAR NOT NULL
        );
        CREATE UNIQUE INDEX ix_users_username ON users (username);

        CREATE TABLE broker_accounts (
            user_id INTEGER PRIMARY KEY REFERENCES users (id) ON DELETE CASCADE,
            ibkr_username VARCHAR(64) NOT NULL,
            ibkr_password VARCHAR NOT NULL,
            account_id VARCHAR(32),
            currency VARCHAR(8),
            net_liquidation NUMERIC,
            cash NUMERIC,
            buying_power NUMERIC,
            available_funds NUMERIC,
            synced_at TIMESTAMPTZ
        );
        CREATE TABLE positions (
            user_id INTEGER NOT NULL REFERENCES broker_accounts (user_id) ON DELETE CASCADE,
            symbol VARCHAR(16) NOT NULL,
            quantity NUMERIC NOT NULL,
            average_cost NUMERIC NOT NULL,
            market_price NUMERIC,
            market_value NUMERIC,
            unrealized_pnl NUMERIC,
            PRIMARY KEY (user_id, symbol)
        );

        CREATE TABLE deployments (
            id BIGSERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users (id) ON DELETE CASCADE,
            thread_id VARCHAR(64) NOT NULL,
            symbol VARCHAR(16) NOT NULL,
            strategy_name VARCHAR(32) NOT NULL,
            parameters JSONB NOT NULL,
            position_tranches INTEGER NOT NULL,
            capital NUMERIC NOT NULL,
            tranches INTEGER NOT NULL DEFAULT 0,
            status VARCHAR(8) NOT NULL DEFAULT 'active',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE TABLE orders (
            id BIGSERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users (id) ON DELETE CASCADE,
            deployment_id BIGINT REFERENCES deployments (id),
            client_order_id UUID NOT NULL,
            symbol VARCHAR(16) NOT NULL,
            side VARCHAR(4) NOT NULL,
            order_type VARCHAR(8) NOT NULL,
            quantity NUMERIC NOT NULL,
            limit_price NUMERIC,
            status VARCHAR(16) NOT NULL DEFAULT 'pending_submit',
            filled_quantity NUMERIC NOT NULL DEFAULT 0,
            avg_fill_price NUMERIC,
            ib_order_id INTEGER,
            error TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (user_id, client_order_id)
        );
        CREATE INDEX ix_orders_user_created ON orders (user_id, created_at DESC);
        CREATE TABLE executions (
            exec_id VARCHAR(64) PRIMARY KEY,
            order_id BIGINT NOT NULL REFERENCES orders (id) ON DELETE CASCADE,
            shares NUMERIC NOT NULL,
            price NUMERIC NOT NULL,
            commission NUMERIC,
            executed_at TIMESTAMPTZ NOT NULL
        );
        CREATE INDEX ix_executions_order ON executions (order_id);

        CREATE TABLE deployment_runs (
            deployment_id BIGINT NOT NULL REFERENCES deployments (id) ON DELETE CASCADE,
            session DATE NOT NULL,
            signal INTEGER,
            tranches INTEGER NOT NULL,
            target_shares INTEGER,
            held_shares NUMERIC NOT NULL,
            order_id BIGINT REFERENCES orders (id),
            note TEXT,
            ran_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (deployment_id, session)
        );

        CREATE TABLE statements (
            id BIGSERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users (id) ON DELETE CASCADE,
            filename VARCHAR(255) NOT NULL,
            object_key VARCHAR(255) NOT NULL,
            imported INTEGER NOT NULL,
            skipped INTEGER NOT NULL,
            uploaded_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE TABLE statement_trades (
            user_id INTEGER NOT NULL REFERENCES users (id) ON DELETE CASCADE,
            statement_id BIGINT NOT NULL REFERENCES statements (id) ON DELETE CASCADE,
            symbol VARCHAR(16) NOT NULL,
            currency VARCHAR(8) NOT NULL,
            traded_at TIMESTAMPTZ NOT NULL,
            quantity NUMERIC NOT NULL,
            price NUMERIC NOT NULL,
            commission NUMERIC NOT NULL,
            UNIQUE (user_id, symbol, traded_at, quantity, price)
        );
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP TABLE statement_trades;
        DROP TABLE statements;
        DROP TABLE deployment_runs;
        DROP TABLE executions;
        DROP TABLE orders;
        DROP TABLE deployments;
        DROP TABLE positions;
        DROP TABLE broker_accounts;
        DROP TABLE users;
        """
    )
