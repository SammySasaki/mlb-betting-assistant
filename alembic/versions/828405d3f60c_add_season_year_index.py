"""Add season year index

Revision ID: 828405d3f60c
Revises: 16867b1a21ef
Create Date: 2025-07-12 18:01:57.463994

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision: str = '828405d3f60c'
down_revision: Union[str, Sequence[str], None] = '16867b1a21ef'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    bind = op.get_bind()
    inspector = inspect(bind)
    indexes = [ix['name'] for ix in inspector.get_indexes('games')]
    if 'idx_games_season_year' not in indexes:
        op.create_index('idx_games_season_year', 'games', ['season_year'])


def downgrade() -> None:
    op.drop_index('idx_games_season_year', table_name='games')