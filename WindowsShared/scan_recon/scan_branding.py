"""Product presentation for the shared local reconstruction engine."""
from pathlib import Path
import base64

BRANDS = {
    "dru": {"id": "dru", "title": "DRU Scan Tools", "product": "Digital ROW Utilities",
            "logo": "dru-logo.png", "prefix": "dru-scan", "action": "#007aff"},
    "bridgecheck": {"id": "bridgecheck", "title": "BridgeCheck AR Scan Tools", "product": "BridgeCheck AR",
                    "logo": "bridgecheck-logo.png", "prefix": "bridgecheck-scan", "action": "#9e4511"},
}


def brand_for(value=None):
    if isinstance(value, dict):
        value = value.get("product") or value.get("producedBy") or value.get("app")
    return BRANDS["bridgecheck" if str(value or "").lower() in
                  ["bridgecheck", "bridgecheck ar", "bridgecheck-ar"] else "dru"]


def logo_uri(brand):
    here = Path(__file__).resolve().parent
    paths = [here / brand["logo"], here.parent / brand["logo"],
             here / "Branding" / brand["logo"], here / "scan_recon" / brand["logo"]]
    path = next((path for path in paths if path.is_file()), None)
    if path is None:
        raise FileNotFoundError("The selected product logo is missing")
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def product_query(product):
    return "?product=bridgecheck" if brand_for(product)["id"] == "bridgecheck" else ""
