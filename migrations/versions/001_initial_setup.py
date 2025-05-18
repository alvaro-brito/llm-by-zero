"""initial setup

Revision ID: 001
Revises: 
Create Date: 2024-03-19 10:00:00.000000

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create enum type for model status if it doesn't exist
    op.execute("""
    DO $$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'modelstatus') THEN
            CREATE TYPE modelstatus AS ENUM (
                'pending',
                'training',
                'completed',
                'failed'
            );
        END IF;
    END
    $$;
    """)
    
    # Create llm_models table
    op.execute("""
    CREATE TABLE IF NOT EXISTS llm_models (
        id SERIAL PRIMARY KEY,
        name VARCHAR NOT NULL,
        description VARCHAR NOT NULL,
        training_data_links VARCHAR NOT NULL,
        status modelstatus DEFAULT 'pending',
        progress FLOAT DEFAULT 0.0,
        model_path VARCHAR,
        error_message VARCHAR,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # Create index if it doesn't exist
    op.execute("""
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1
            FROM pg_indexes
            WHERE tablename = 'llm_models'
            AND indexname = 'ix_llm_models_id'
        ) THEN
            CREATE INDEX ix_llm_models_id ON llm_models (id);
        END IF;
    END
    $$;
    """)


def downgrade() -> None:
    # Drop everything in reverse order
    op.execute("DROP INDEX IF EXISTS ix_llm_models_id;")
    op.execute("DROP TABLE IF EXISTS llm_models CASCADE;")
    op.execute("DROP TYPE IF EXISTS modelstatus CASCADE;") 