"""Atomic, content-addressed inference checkpoints and shared utilities."""

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def extract_choice(text: str, choices: list[str]) -> str | None:
    """Accept an explicitly marked option; do not infer it from scratch reasoning."""
    matches = list(
        re.finditer(r"Final answer\s*:\s*(?:\*\*)?\s*\(?([A-Z])\)?(?![A-Za-z0-9])", text, re.I)
    )
    if not matches:
        return None
    value = matches[-1].group(1).upper()
    return value if value in choices else None


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def source_files():
    package = Path(__file__).resolve().parent
    return {
        "dirty_swapping/" + str(p.relative_to(package)): p
        for p in sorted(package.rglob("*"))
        if p.suffix in (".py", ".json") and "__pycache__" not in p.parts
    }


def source_fingerprint():
    return {name: file_digest(path) for name, path in source_files().items()}
