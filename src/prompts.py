"""Construction des prompts (identiques pour baseline et modèle fine-tuné)."""
from __future__ import annotations

import json

from .config import FIELDS

SYSTEM_PROMPT = (
    "You are an information extraction system for scanned receipts. "
    "Given the OCR text of a receipt, return ONLY a JSON object with exactly these keys: "
    '"company", "date", "address", "total".\n'
    "Rules:\n"
    "- company: the name of the shop/company that issued the receipt (usually near the top, "
    "often ending with SDN BHD, ENTERPRISE, TRADING...). Not the customer.\n"
    "- date: the transaction date, copied as written on the receipt, without the time.\n"
    "- address: the full address of the company, all lines joined with a space, "
    "without phone/fax/email/GST numbers.\n"
    "- total: the final amount paid (after tax and rounding), digits only, e.g. \"12.50\", "
    "no currency symbol.\n"
    "- Copy values exactly from the text. Never invent a value: if a field is absent, use null.\n"
    "- Output the JSON object only, no explanation."
)


def user_message(text: str) -> str:
    return f"Receipt OCR text:\n<<<\n{text}\n>>>"


def target_json(target: dict) -> str:
    return json.dumps({f: (target.get(f) or None) for f in FIELDS}, ensure_ascii=False)


def build_messages(text: str, demos: list[dict] | None = None) -> list[dict]:
    """demos : exemples (few-shot fixe ou récupérés par le RAG) au format
    {text, target}, insérés comme tours user/assistant précédents."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for d in demos or []:
        msgs.append({"role": "user", "content": user_message(d["text"])})
        msgs.append({"role": "assistant", "content": target_json(d["target"])})
    msgs.append({"role": "user", "content": user_message(text)})
    return msgs
