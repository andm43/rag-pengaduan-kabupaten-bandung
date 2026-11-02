import os
import re
import warnings

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")
warnings.filterwarnings("ignore", message=".*unauthenticated.*")
warnings.filterwarnings("ignore", message=".*sequentially on GPU.*")

import pandas as pd

_ner_pipeline = None
_ner_checked = False

def _get_ner():
    """Lazy-load NER pipeline. Return None jika transformers tidak ada."""
    global _ner_pipeline, _ner_checked
    if _ner_checked:
        return _ner_pipeline
    _ner_checked = True
    try:
        import logging
        for name in ("transformers", "transformers.modeling_utils",
                     "transformers.pipelines", "transformers.configuration_utils"):
            logging.getLogger(name).setLevel(logging.CRITICAL)
        from transformers import pipeline as hf_pipeline
        import contextlib, io
        with contextlib.redirect_stderr(io.StringIO()):
            _ner_pipeline = hf_pipeline(
                "ner",
                model="cahya/bert-base-indonesian-NER",
                aggregation_strategy="simple",
            )
    except Exception:
        _ner_pipeline = None
    return _ner_pipeline


NAME_STOPWORDS = (
    r"(?!(?:saya|aku|kami|kita|dengan|dan|yang|di|ke|dari|pada|untuk|"
    r"mohon|ingin|mau|akan|adalah|ialah|selaku|sebagai|warga|penduduk|"
    r"beralamat|alamat|tinggal|umur|usia|sudah|belum|telah|tidak|bukan|"
    r"melaporkan|melapor|lapor|mengadukan|menanyakan|bermaksud|asal|"
    r"lahir|nomor|nik|kk|hp|email|"
    r"bupati|wakil|gubernur|presiden|menteri|camat|lurah|kades|kepala|"
    r"dinas|badan|kantor|desa|kecamatan|kabupaten|pemerintah|pemkab|"
    r"bapak|ibu|bpk|sdr|sdri|saudara|saudari|pak|bu)\b)"
)

HONORIFIC = r"(?:bapak|ibu|bpk\.?|sdr\.?i?|saudara|saudari|pak|bu)\s+"

NAME_TOKENS = (
    rf"{NAME_STOPWORDS}[a-z][a-z'().\-]*"
    rf"(?:\s+{NAME_STOPWORDS}[a-z][a-z'().\-]*){{0,4}}"
)

NAME_LABEL_BOUNDARY = r"(?<!ber)"

TEXT_COLUMNS = ("judul", "isi_laporan", "tindak_lanjut")

_NAME_LABEL_PATTERN = re.compile(
    rf"{NAME_LABEL_BOUNDARY}"
    rf"(nama(?:\s+(?:saya|pelapor|lengkap|wp))?\s*:\s*(?:{HONORIFIC})?)"
    rf"({NAME_TOKENS})",
    re.IGNORECASE,
)
_AN_LABEL_PATTERN = re.compile(
    rf"{NAME_LABEL_BOUNDARY}(a\.?\s*n\.?\s*:\s*(?:{HONORIFIC})?)"
    rf"({NAME_TOKENS})",
    re.IGNORECASE,
)
_NAME_NARRATIVE_PATTERNS = [
    re.compile(rf"(\bnama\s+saya\s+(?:{HONORIFIC})?)({NAME_TOKENS})", re.IGNORECASE),
    re.compile(rf"(\bbernama\s+(?:{HONORIFIC})?)({NAME_TOKENS})", re.IGNORECASE),
    re.compile(rf"(\batas\s+nama\s+(?:{HONORIFIC})?)({NAME_TOKENS})", re.IGNORECASE),
    re.compile(rf"(\bsaya\s+)({NAME_TOKENS})(?=\s+dari\b)", re.IGNORECASE),
    re.compile(rf"(\bpelapor\s*:\s*(?:{HONORIFIC})?)({NAME_TOKENS})", re.IGNORECASE),
    re.compile(rf"(\byth\.?\s+(?:{HONORIFIC})?)({NAME_TOKENS})", re.IGNORECASE),
    re.compile(
        rf"(?<!ber)(nama\s+)({NAME_TOKENS})(?=\s+(?:no|ksp|nik)\b)",
        re.IGNORECASE,
    ),
]


def _to_initials(raw_name):
    tokens = [
        token for token in re.split(r"\s+", str(raw_name).strip())
        if token and re.match(r"^[a-zA-Z]", token)
    ]
    if not tokens:
        return "[NAMA]"
    initials = [token[0].upper() for token in tokens[:4]]
    return ".".join(initials) + "."


def _replace_name_label(match):
    prefix = match.group(1)
    raw_name = match.group(2)
    return f"{prefix}{_to_initials(raw_name)}"


def anonymize_text(text):

    if pd.isna(text):
        return ""

    text = str(text)


    text = re.sub(r"(?<!\d)08\d{8,12}(?!\d)", "[PHONE]", text)

    text = re.sub(r"(?<!\d)\d{16}(?!\d)", "[NIK]", text)

    text = re.sub(
        r"(?<!\d)\d{2}\.\d{2}\.\d{3}\.\d{3}\.\d{3}\.\d{4}\.\d(?!\d)",
        "[NOP]",
        text,
    )

    text = re.sub(
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "[EMAIL]",
        text,
    )

    text = re.sub(
        r"\b[a-zA-Z0-9._%+-]+\s+(?:gmail|yahoo|hotmail|outlook)\.com\b",
        "[EMAIL]",
        text,
    )

    text = _NAME_LABEL_PATTERN.sub(_replace_name_label, text)
    text = _AN_LABEL_PATTERN.sub(_replace_name_label, text)

    for pattern in _NAME_NARRATIVE_PATTERNS:
        text = pattern.sub(_replace_name_label, text)

    ner = _get_ner()
    if ner is not None:
        try:
            entities = ner(text)
            for ent in sorted(entities, key=lambda e: e["start"], reverse=True):
                if ent["entity_group"] != "PER" or ent["score"] < 0.85:
                    continue
                span = text[ent["start"]:ent["end"]].strip()
                if len(span) < 3 or span.startswith("["):
                    continue
                text = text[:ent["start"]] + _to_initials(span) + text[ent["end"]:]
        except Exception:
            pass

    return text


def anonymize_dataframe(df, columns=TEXT_COLUMNS):
    """Anonimkan seluruh kolom teks sekaligus.

    Dipanggil satu kali tepat setelah CSV dibaca, sebelum row dipakai
    untuk apa pun. Ini choke point-nya: dengan begini isi chunk MAUPUN
    metadata YAML sama-sama bersih, tidak ada jalur yang terlewat.
    """
    df = df.copy()

    for column in columns:
        if column in df.columns:
            df[column] = df[column].map(anonymize_text)

    return df


PII_SCANNERS = (
    ("NIK", re.compile(r"(?<!\d)\d{16}(?!\d)")),
    ("PHONE", re.compile(r"(?<!\d)08\d{8,12}(?!\d)")),
    ("EMAIL", re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")),
    (
        "NAMA",
        re.compile(
            rf"(?i){NAME_LABEL_BOUNDARY}"
            rf"nama(?:\s+(?:saya|pelapor|lengkap))?\s*:\s*"
            rf"(?:{HONORIFIC})?{NAME_STOPWORDS}[a-z]{{3,}}"
        ),
    ),
)


def scan_pii(text):
    """Kembalikan daftar (jenis, cuplikan) sisa PII yang terdeteksi."""
    findings = []

    if not text:
        return findings

    for label, pattern in PII_SCANNERS:
        for match in pattern.finditer(str(text)):
            findings.append((label, match.group(0)))

    return findings