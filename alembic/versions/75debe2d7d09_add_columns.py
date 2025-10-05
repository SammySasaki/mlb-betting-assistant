"""Add columns

Revision ID: 75debe2d7d09
Revises: 4bf60802dad6
Create Date: 2025-07-12 16:59:31.043959

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '75debe2d7d09'
down_revision: Union[str, Sequence[str], None] = '4bf60802dad6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
