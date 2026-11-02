"""Orchestrator for the preprocessing → corpus → indexing pipeline.

Called from the Streamlit dashboard to let staff run the full pipeline
without touching code. Each stage delegates to the existing scripts
(preprocessing.py, build_corpus_aduan.py, build_chroma_index.py,
build_bm25_index.py) so behaviour stays identical to the CLI path.

Design: thin wrapper, no duplicated logic. Each function captures stdout
for the dashboard log panel.
"""

import contextlib
import io
import os
import shutil
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
DATASET_DIR.mkdir(exist_ok=True)


def _capture(fn, *args, **kwargs):
    """Run fn, capture its stdout, return (result, log_text)."""
    buf = io.StringIO()
    result = None
    with contextlib.redirect_stdout(buf):
        result = fn(*args, **kwargs)
    return result, buf.getvalue()


def save_uploaded_excel(uploaded_files: list, overwrite: bool = False):
    """Save Streamlit UploadedFile objects into dataset/.
    Returns list of (filename, status) tuples."""
    results = []
    for uf in uploaded_files:
        dest = DATASET_DIR / uf.name
        if dest.exists() and not overwrite:
            results.append((uf.name, "sudah ada (skip)"))
            continue
        dest.write_bytes(uf.getvalue())
        results.append((uf.name, "tersimpan"))
    return results


def list_dataset_files():
    """Return sorted list of .xlsx files in dataset/."""
    return sorted(p.name for p in DATASET_DIR.glob("*.xlsx"))


def run_preprocessing():
    """Run preprocessing.main(). Returns log text."""
    import preprocessing
    current_files = list_dataset_files()
    if not current_files:
        return "Tidak ada file .xlsx di folder dataset/"

    original_files = preprocessing.FILES
    preprocessing.FILES = current_files
    try:
        _, log = _capture(preprocessing.main)
    finally:
        preprocessing.FILES = original_files
    return log


def run_build_corpus_aduan():
    """Run build_corpus_aduan.main(). Returns log text."""
    import build_corpus_aduan
    _, log = _capture(build_corpus_aduan.main)
    return log


def run_build_corpus_sop():
    """Run build_corpus_sop.main(). Returns log text."""
    import build_corpus_sop
    _, log = _capture(build_corpus_sop.main)
    return log


def run_build_chroma_index():
    """Run build_chroma_index.main(). Returns log text."""
    import build_chroma_index
    _, log = _capture(build_chroma_index.main)
    return log


def run_build_bm25_index():
    """Run build_bm25_index.main(). Returns log text."""
    import build_bm25_index
    _, log = _capture(build_bm25_index.main)
    return log


REGULASI_DIR = BASE_DIR / "regulasi_uploads"
REGULASI_DIR.mkdir(exist_ok=True)


def save_uploaded_pdfs(uploaded_files: list, overwrite: bool = False):
    """Save Streamlit UploadedFile PDFs into regulasi_uploads/.
    Returns list of (filename, status) tuples."""
    results = []
    for uf in uploaded_files:
        dest = REGULASI_DIR / uf.name
        if dest.exists() and not overwrite:
            results.append((uf.name, "sudah ada (skip)"))
            continue
        dest.write_bytes(uf.getvalue())
        results.append((uf.name, "tersimpan"))
    return results


def list_regulasi_files():
    """Return sorted list of .pdf files in regulasi_uploads/."""
    return sorted(p.name for p in REGULASI_DIR.glob("*.pdf"))


def run_extract_and_build_corpus_regulasi(progress_callback=None):
    """Extract all PDFs in regulasi_uploads/ via GPT-4o vision,
    then chunk and write to corpus_regulasi_sop/.

    Falls back to build_corpus_sop.main() if no PDFs uploaded.
    """
    pdf_files = list(REGULASI_DIR.glob("*.pdf"))

    if not pdf_files:
        print("[INFO] Tidak ada PDF di regulasi_uploads/, gunakan corpus SOP hardcoded.")
        return run_build_corpus_sop()

    from extract_pdf import extract_pdf
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import re

    output_dir = BASE_DIR / "corpus_regulasi_sop"
    output_dir.mkdir(exist_ok=True)

    for old in output_dir.glob("*.md"):
        old.unlink()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=False,
    )

    chunk_counter = 0

    for pdf_path in sorted(pdf_files):
        doc_name = pdf_path.stem
        source_id = re.sub(r"[^a-z0-9]+", "-", doc_name.lower()).strip("-") or "regulasi"

        print(f"\n[EXTRACT] {pdf_path.name}")
        pages = extract_pdf(pdf_path, progress_callback=progress_callback)

        for page_data in pages:
            page_num = page_data["page"]
            text = page_data["text"].strip()

            if not text or text == "[HALAMAN KOSONG]":
                continue

            chunks = splitter.split_text(text)

            for ci, chunk_text in enumerate(chunks):
                chunk_text = chunk_text.strip()
                if not chunk_text:
                    continue

                chunk_id = f"{source_id}-p{page_num:03d}-{ci:02d}"
                filename = f"regulasi_{chunk_counter:04d}_{chunk_id}.md"

                def _yaml_quote(v):
                    v = str(v).replace("\\", "\\\\").replace('"', '\\"')
                    return f'"{v}"'

                md = f"""---
source_id: {_yaml_quote(source_id)}
chunk_id: {_yaml_quote(chunk_id)}
jenis: {_yaml_quote("regulasi")}
tipe_dokumen: {_yaml_quote("sop")}
dokumen: {_yaml_quote(doc_name)}
sumber_file: {_yaml_quote(pdf_path.name)}
halaman: {_yaml_quote(str(page_num))}
section: {_yaml_quote(f"Halaman {page_num}")}
---

{chunk_text}
"""
                (output_dir / filename).write_text(md, encoding="utf-8")
                chunk_counter += 1

        print(f"[OK] {pdf_path.name}: {len(pages)} halaman diekstrak")

    print(f"\n[DONE] Total chunk regulasi/SOP: {chunk_counter}")
    return None


STAGES = [
    ("Pembersihan Data", run_preprocessing),
    ("Bangun Korpus Aduan", run_build_corpus_aduan),
    ("Bangun Korpus Regulasi/SOP", run_extract_and_build_corpus_regulasi),
    ("Bangun Indeks Vektor", run_build_chroma_index),
    ("Bangun Indeks Kata Kunci", run_build_bm25_index),
]


def run_full_pipeline(progress_callback=None):
    """Run all stages sequentially.

    progress_callback(stage_index, stage_name, status, log) is called
    after each stage completes. status is 'ok' or 'error'.

    Returns list of (stage_name, status, log, elapsed_seconds).
    """
    results = []
    for i, (name, fn) in enumerate(STAGES):
        t0 = time.time()
        try:
            log = fn()
            status = "ok"
        except Exception as exc:
            log = f"ERROR: {exc}"
            status = "error"
        elapsed = time.time() - t0

        results.append((name, status, log, elapsed))
        if progress_callback:
            progress_callback(i, name, status, log)

        if status == "error":
            break

    return results


def get_index_stats():
    """Return dict with corpus and index file stats."""
    stats = {}

    corpus_aduan = BASE_DIR / "corpus_aduan"
    corpus_sop = BASE_DIR / "corpus_regulasi_sop"
    chroma_dir = BASE_DIR / "chroma_rag"
    bm25_path = chroma_dir / "bm25_index.pkl"
    csv_path = BASE_DIR / "dataset_clean.csv"

    stats["corpus_aduan_chunks"] = sum(1 for _ in corpus_aduan.glob("*.md")) if corpus_aduan.exists() else 0
    stats["corpus_sop_chunks"] = sum(1 for _ in corpus_sop.glob("*.md")) if corpus_sop.exists() else 0
    stats["total_chunks"] = stats["corpus_aduan_chunks"] + stats["corpus_sop_chunks"]
    stats["chroma_exists"] = chroma_dir.exists() and any(chroma_dir.iterdir()) if chroma_dir.exists() else False
    stats["bm25_exists"] = bm25_path.exists()
    stats["bm25_size_mb"] = round(bm25_path.stat().st_size / 1e6, 1) if bm25_path.exists() else 0
    stats["csv_exists"] = csv_path.exists()

    if csv_path.exists():
        import pandas as pd
        try:
            stats["csv_rows"] = len(pd.read_csv(csv_path))
        except Exception:
            stats["csv_rows"] = 0
    else:
        stats["csv_rows"] = 0

    stats["dataset_files"] = list_dataset_files()
    stats["regulasi_files"] = list_regulasi_files()

    return stats


import subprocess
import signal

_bot_process = None


def start_bot():
    """Start telegram_bot.py as a subprocess. Returns (ok, message)."""
    global _bot_process
    if _bot_process is not None and _bot_process.poll() is None:
        return False, "Bot sudah berjalan (PID {})".format(_bot_process.pid)

    bot_script = BASE_DIR / "telegram_bot.py"
    if not bot_script.exists():
        return False, "telegram_bot.py tidak ditemukan"

    try:
        _bot_process = subprocess.Popen(
            ["python", str(bot_script)],
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return True, "Bot dimulai (PID {})".format(_bot_process.pid)
    except Exception as exc:
        return False, "Gagal memulai bot: {}".format(exc)


def stop_bot():
    """Stop the running bot subprocess. Returns (ok, message)."""
    global _bot_process
    if _bot_process is None or _bot_process.poll() is not None:
        _bot_process = None
        return False, "Bot tidak sedang berjalan"

    try:
        _bot_process.terminate()
        _bot_process.wait(timeout=10)
        pid = _bot_process.pid
        _bot_process = None
        return True, "Bot dihentikan (PID {})".format(pid)
    except subprocess.TimeoutExpired:
        _bot_process.kill()
        _bot_process = None
        return True, "Bot di-force-kill"
    except Exception as exc:
        return False, "Gagal menghentikan bot: {}".format(exc)


def bot_status():
    """Return (running: bool, info: str)."""
    global _bot_process
    if _bot_process is None:
        return False, "Tidak aktif"
    if _bot_process.poll() is None:
        return True, "Aktif (PID {})".format(_bot_process.pid)
    code = _bot_process.returncode
    _bot_process = None
    return False, "Berhenti (exit code {})".format(code)
