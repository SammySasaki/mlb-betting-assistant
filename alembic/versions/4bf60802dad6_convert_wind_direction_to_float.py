"""Convert wind_direction to Float

Revision ID: 4bf60802dad6
Revises: b1a647607e16
Create Date: 2025-07-12 15:59:37.927414

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4bf60802dad6'
down_revision: Union[str, Sequence[str], None] = 'b1a647607e16'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # First, convert existing text values to numeric (if needed)
    op.execute("""
        UPDATE weather
        SET wind_direction = NULL
        WHERE wind_direction !~ '^[0-9]+(\\.[0-9]+)?$'
    """)

    # Alter the column from String to Float
    op.alter_column('weather', 'wind_direction',
        existing_type=sa.String(),
        type_=sa.Float(),
        postgresql_using='wind_direction::double precision'
    )


def downgrade():
    # Revert back to String
    op.alter_column('weather', 'wind_direction',
        existing_type=sa.Float(),
        type_=sa.String(),
        postgresql_using='wind_direction::text'
    )