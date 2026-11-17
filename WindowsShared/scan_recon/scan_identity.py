"""Capture naming and the time zone recorded by the capture device."""
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def identity(manifest):
    name = next((str(manifest[key]).strip() for key in ["scanName", "permitNumber", "projectName"] if manifest.get(key) and str(manifest[key]).strip()), "Scan " + str(manifest.get("id", "unknown"))[:8])
    created = manifest.get("createdAt") or manifest.get("capturedUTC") or ""
    zone_name = manifest.get("captureTimeZone")
    offset = manifest.get("captureUTCOffsetSeconds")
    try:
        instant = datetime.fromisoformat(created.replace("Z", "+00:00"))
        if instant.tzinfo is None:
            raise ValueError("Capture instant has no offset")
        zone = timezone.utc
        if isinstance(offset, (int, float)) and -64800 <= offset <= 64800:
            zone = timezone(timedelta(seconds=offset))
        elif zone_name:
            zone = ZoneInfo(zone_name)
        local = instant.astimezone(zone)
        date = local.strftime("%b %d, %Y · %I:%M %p UTC%z")
        date += " · " + zone_name if zone_name else (" · capture zone unavailable" if offset is None else "")
    except (ValueError, TypeError, ZoneInfoNotFoundError):
        date = created or "Capture time unavailable"
    return {"name": name, "date": date, "createdAt": created,
            "captureTimeZone": zone_name, "captureUTCOffsetSeconds": offset}
