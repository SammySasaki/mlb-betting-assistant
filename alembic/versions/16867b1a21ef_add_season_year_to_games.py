"""Add season_year to games

Revision ID: 16867b1a21ef
Revises: 7022157f5b75
Create Date: 2025-07-12 17:57:25.548306

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '16867b1a21ef'
down_revision: Union[str, Sequence[str], None] = '7022157f5b75'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.add_column('games', sa.Column('season_year', sa.Integer))
    op.execute("UPDATE games SET season_year = EXTRACT(YEAR FROM date)")
    op.create_index('idx_games_season_year', 'games', ['season_year'])

def downgrade():
    op.drop_index('idx_games_season_year', table_name='games')
    op.drop_column('games', 'season_year')
