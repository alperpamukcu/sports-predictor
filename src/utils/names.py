from unidecode import unidecode

def normalize_name(s: str | None) -> str:
    if not s:
        return ""
    s = unidecode(s).strip().lower()
    return " ".join(s.split())
