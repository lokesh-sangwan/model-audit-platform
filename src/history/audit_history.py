import json
from pathlib import Path
from datetime import datetime


# ============================================================
# STORAGE CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIRECTORY = PROJECT_ROOT / "data"

HISTORY_FILE = DATA_DIRECTORY / "audit_history.json"


# ============================================================
# STORAGE INITIALIZATION
# ============================================================

def _ensure_storage_exists():
    """
    Ensures that the local data directory and audit-history
    file exist.
    """

    DATA_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    if not HISTORY_FILE.exists():

        HISTORY_FILE.write_text(
            "[]",
            encoding="utf-8"
        )


# ============================================================
# LOAD HISTORY
# ============================================================

def load_audit_history():
    """
    Loads all previously saved audit records.

    Returns:
        List of audit records.
    """

    _ensure_storage_exists()

    try:

        content = HISTORY_FILE.read_text(
            encoding="utf-8"
        )

        return json.loads(
            content
        )

    except (
        json.JSONDecodeError,
        OSError
    ):

        return []


# ============================================================
# SAVE HISTORY
# ============================================================

def save_audit_record(
    audit_report
):
    """
    Saves a completed audit report to local history.

    Args:
        audit_report: Structured audit report dictionary.

    Returns:
        Unique audit identifier.
    """

    history = load_audit_history()

    audit_id = (
        datetime.now()
        .strftime("%Y%m%d%H%M%S%f")
    )

    record = {

        "audit_id": audit_id,

        "audit_report": audit_report

    }

    history.append(
        record
    )

    HISTORY_FILE.write_text(
        json.dumps(
            history,
            indent=4
        ),
        encoding="utf-8"
    )

    return audit_id


# ============================================================
# CHECK FOR DUPLICATE CURRENT AUDIT
# ============================================================

def audit_already_saved(
    audit_report
):
    """
    Checks whether the same audit report has already been
    stored in history.

    This prevents repeated page refreshes from creating
    duplicate audit records.
    """

    history = load_audit_history()

    for record in history:

        if record.get(
            "audit_report"
        ) == audit_report:

            return True

    return False