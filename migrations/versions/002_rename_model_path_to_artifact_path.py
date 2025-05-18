"""rename model_path to artifact_path

Revision ID: 002
Revises: 001
Create Date: 2024-03-19

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None

def upgrade():
    # Rename model_path column to artifact_path
    op.alter_column('llm_models', 'model_path',
                    new_column_name='artifact_path',
                    existing_type=sa.String(),
                    nullable=True)

def downgrade():
    # Rename artifact_path column back to model_path
    op.alter_column('llm_models', 'artifact_path',
                    new_column_name='model_path',
                    existing_type=sa.String(),
                    nullable=True) 