from __future__ import annotations

import hashlib
import re
import unicodedata


URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+|\b[a-z0-9.-]+\.(?:com|net|org|io|co|cn|me|info|biz)\b\S*")
EMAIL_RE = re.compile(r"(?i)\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\w)\+?[\d() .-]{7,}\d(?!\w)")
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize representation without erasing signals useful for spam."""
    from .native import normalize
    return normalize(str(text))



def template_text(text: str) -> str:
    """Collapse campaign variables so a template cannot leak across splits."""
    from .native import normalize
    text = normalize(text, fold=True)
    text = URL_RE.sub(" <url> ", text)
    text = EMAIL_RE.sub(" <email> ", text)
    text = PHONE_RE.sub(" <phone> ", text)
    text = NUMBER_RE.sub(" <num> ", text)
    text = "".join(ch if (ch.isalnum() or ch in "<> ") else " " for ch in text)
    return SPACE_RE.sub(" ", text).strip()


def stable_hash(value: str, *, salt: str = "potasms-v1") -> str:
    return hashlib.sha256(f"{salt}\0{value}".encode("utf-8")).hexdigest()


def split_for_template(template: str) -> str:
    bucket = int(stable_hash(template)[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"

