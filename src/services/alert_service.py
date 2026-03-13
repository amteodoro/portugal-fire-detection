"""Fire alert email service.

Checks AOI subscriptions against recent fire events and sends email alerts.
Only runs when DATABASE_URL and SMTP_HOST are both configured.
"""
from __future__ import annotations

import logging
import smtplib
from datetime import datetime, timezone, timedelta
from email.message import EmailMessage

from src.utils.config import settings

logger = logging.getLogger(__name__)


def is_configured() -> bool:
    """Returns True only when both DB and SMTP are configured."""
    from src.db.session import is_configured as db_ok

    return db_ok() and bool(settings.SMTP_HOST)


def _send_email(to: str, subject: str, body: str) -> None:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = settings.FROM_EMAIL
    msg["To"] = to
    msg.set_content(body)

    with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as smtp:
        smtp.ehlo()
        if settings.SMTP_PORT == 587:
            smtp.starttls()
        if settings.SMTP_USER and settings.SMTP_PASS:
            smtp.login(settings.SMTP_USER, settings.SMTP_PASS)
        smtp.send_message(msg)


async def check_and_alert() -> int:
    """Main alert job — returns number of alerts sent."""
    if not is_configured():
        return 0

    from sqlalchemy import text
    from src.db.session import get_session

    alerts_sent = 0

    # Find AOIs with email subscriptions
    aoi_sql = text(
        "SELECT id, name, email FROM areas_of_interest"
        " WHERE active = true AND email IS NOT NULL AND email != ''"
    )

    try:
        async with get_session() as session:
            aoi_result = await session.execute(aoi_sql)
            aois = aoi_result.fetchall()
    except Exception as exc:
        logger.error("Alert job: failed to fetch AOIs: %s", exc)
        return 0

    for aoi in aois:
        # Find new fire events within this AOI since last alert
        fires_sql = text(
            "SELECT fe.id, fe.source, fe.detected_at, fe.frp,"
            " ST_Y(fe.location::geometry) AS lat, ST_X(fe.location::geometry) AS lon"
            " FROM fire_events fe"
            " JOIN areas_of_interest a ON a.id = :aoi_id"
            " WHERE ST_Within(fe.location::geometry, a.boundary::geometry)"
            "   AND fe.detected_at >= NOW() - INTERVAL '30 minutes'"
            "   AND fe.id NOT IN ("
            "     SELECT event_id FROM alert_log"
            "     WHERE aoi_id = :aoi_id AND event_id IS NOT NULL"
            "   )"
            " ORDER BY fe.detected_at DESC LIMIT 50"
        )
        try:
            async with get_session() as session:
                result = await session.execute(fires_sql, {"aoi_id": str(aoi.id)})
                new_fires = result.fetchall()
        except Exception as exc:
            logger.error("Alert job: fires query failed for AOI %s: %s", aoi.id, exc)
            continue

        if not new_fires:
            continue

        # Build email
        area_name = aoi.name or f"AOI {str(aoi.id)[:8]}"
        fire_lines = [
            f"  - {f.source} at ({f.lat:.4f}, {f.lon:.4f})"
            f" | FRP: {f.frp:.1f} MW" if f.frp else f"  - {f.source} at ({f.lat:.4f}, {f.lon:.4f})"
            for f in new_fires
        ]
        body = (
            f"Fire Alert — {area_name}\n\n"
            f"{len(new_fires)} new fire detection(s) in your area of interest "
            f"in the last 30 minutes:\n\n"
            + "\n".join(fire_lines)
            + "\n\nThis is an automated alert from Portugal Fire Detection.\n"
        )
        try:
            _send_email(
                to=aoi.email,
                subject=f"[Fire Alert] {len(new_fires)} new detection(s) in {area_name}",
                body=body,
            )
            logger.info(
                "Alert sent to %s for AOI %s (%d fires)", aoi.email, aoi.id, len(new_fires)
            )
        except Exception as exc:
            logger.error("Alert email failed for %s: %s", aoi.email, exc)
            continue

        # Log alerts to avoid resending
        try:
            async with get_session() as session:
                for fire in new_fires:
                    from src.db.models import AlertLog

                    session.add(
                        AlertLog(
                            aoi_id=aoi.id,
                            event_id=fire.id,
                            sent_at=datetime.now(timezone.utc),
                        )
                    )
                await session.commit()
        except Exception as exc:
            logger.error("Alert log failed for AOI %s: %s", aoi.id, exc)

        alerts_sent += len(new_fires)

    return alerts_sent
