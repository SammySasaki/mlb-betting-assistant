"""Add indexes for performance

Revision ID: 7022157f5b75
Revises: 2caba8a9784a
Create Date: 2025-07-12 17:54:00.739709

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7022157f5b75'
down_revision: Union[str, Sequence[str], None] = '2caba8a9784a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():

    # Indexes on player_game_stats
    op.create_index('idx_pgs_player_game', 'player_game_stats', ['player_id', 'game_id'])
    op.create_index('idx_pgs_team_date', 'player_game_stats', ['team', 'game_id'])

    # Index on games.date
    op.create_index('idx_games_date', 'games', ['date'])

def downgrade():
    op.drop_index('idx_games_season_year', table_name='games')
    op.drop_index('idx_pgs_player_game', table_name='player_game_stats')
    op.drop_index('idx_pgs_team_date', table_name='player_game_stats')
    op.drop_index('idx_games_date', table_name='games')
