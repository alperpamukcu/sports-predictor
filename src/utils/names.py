from unidecode import unidecode
def normalize_name(s: str | None) -> str:
    if not s:
        return ""
    s = unidecode(s)
    s = s.strip().lower()
    return " ".join(s.split())

if __name__ == "__main__":
    samples = ["Daniil Medvédev", "  Rafael  Nadal ", None, "İGA Świątek"]
    for t in samples:
        print(t, "->", normalize_name(t))
