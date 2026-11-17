"""Use the native Keychain and signature verifier before local processing."""
import json
import subprocess


def access_status(config):
    if config.get("access_platform") == "windows":
        try:
            from windows_access import status
            return status(config["access_configuration"], config["access_state"])
        except Exception:
            return {"allowed": False, "kind": "unavailable", "message": "Open Account in Scan Tools to check your desktop license."}
    helper = config.get("access_helper")
    if helper is None:
        return {"allowed": True, "kind": "internal", "message": "Local scan worker is ready."}
    try:
        result = subprocess.run(
            [helper, "status", "--config", config["access_configuration"]],
            check=True, capture_output=True, text=True, timeout=30,
        )
        value = json.loads(result.stdout)
        if not isinstance(value.get("allowed"), bool):
            raise ValueError("Missing license status")
        return value
    except (OSError, subprocess.SubprocessError, ValueError, KeyError):
        return {"allowed": False, "kind": "unavailable", "message": "Open the Mac companion to check your desktop license."}


def require_access(config):
    status = access_status(config)
    if not status["allowed"]:
        raise PermissionError(status["message"])
