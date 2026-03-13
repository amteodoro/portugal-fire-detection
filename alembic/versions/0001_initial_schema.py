"""Initial schema: fire_events, areas_of_interest, alert_log

Revision ID: 0001
Revises:
Create Date: 2026-03-13
"""
from typing import Sequence, Union

import geoalchemy2
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis")

    op.create_table(
        "fire_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("source", sa.String(10), nullable=False),
        sa.Column("detected_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "location",
            geoalchemy2.Geography("POINT", srid=4326),
            nullable=False,
        ),
        sa.Column("frp", sa.Float(), nullable=True),
        sa.Column("confidence", sa.String(10), nullable=True),
        sa.Column("brightness", sa.Float(), nullable=True),
        sa.Column("region", sa.String(50), nullable=True),
    )
    op.create_index(
        "ix_fire_events_location",
        "fire_events",
        ["location"],
        postgresql_using="gist",
    )
    op.create_index("ix_fire_events_detected_at", "fire_events", ["detected_at"])
    op.create_index("ix_fire_events_source", "fire_events", ["source"])

    op.create_table(
        "areas_of_interest",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("name", sa.String(100), nullable=True),
        sa.Column("email", sa.String(255), nullable=True),
        sa.Column(
            "boundary",
            geoalchemy2.Geography("POLYGON", srid=4326),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.Column("active", sa.Boolean(), default=True, server_default="true"),
    )
    op.create_index(
        "ix_aoi_boundary",
        "areas_of_interest",
        ["boundary"],
        postgresql_using="gist",
    )

    op.create_table(
        "alert_log",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "aoi_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("areas_of_interest.id"),
            nullable=True,
        ),
        sa.Column(
            "event_id",
            sa.Integer(),
            sa.ForeignKey("fire_events.id"),
            nullable=True,
        ),
        sa.Column(
            "sent_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )


def downgrade() -> None:
    op.drop_table("alert_log")
    op.drop_table("areas_of_interest")
    op.drop_table("fire_events")
