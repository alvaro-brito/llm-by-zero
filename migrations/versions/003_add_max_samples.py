"""add max_samples column

Revision ID: 003
Revises: 002
Create Date: 2024-03-19

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic
revision = '003'
down_revision = '002'
branch_labels = None
depends_on = None

def upgrade():
    # Add max_samples column
    op.add_column('llm_models', sa.Column('max_samples', sa.Integer(), nullable=True))

def downgrade():
    # Remove max_samples column
    op.drop_column('llm_models', 'max_samples') 