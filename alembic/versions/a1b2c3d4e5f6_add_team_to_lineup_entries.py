"""add team to lineup_entries

Revision ID: a1b2c3d4e5f6
Revises: fe5911ce7363
Create Date: 2026-08-15 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = 'fe5911ce7363'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('lineup_entries', sa.Column('team', sa.String(), nullable=True))
    op.drop_constraint('uq_lineup_slot', 'lineup_entries', type_='unique')
    op.create_unique_constraint('uq_lineup_slot', 'lineup_entries', ['game_id', 'batting_order', 'team'])


def downgrade() -> None:
    op.drop_constraint('uq_lineup_slot', 'lineup_entries', type_='unique')
    op.create_unique_constraint('uq_lineup_slot', 'lineup_entries', ['game_id', 'batting_order'])
    op.drop_column('lineup_entries', 'team')
