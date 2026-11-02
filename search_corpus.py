import argparse
import ctypes
import json
import os
import re
import sys
from pathlib import Path

import yaml


BASE_DIR = Path(__file__).resolve().parent
CORPUS_GLOB = "corpus_*"
CONTEXT_CHARS = 90
MAX_RESULTS = 20
MAX_PER_FILE = 3


def enable_ansi_on_windows():
    if os.name != "nt":
        return

    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass


def use_color(no_color):
    if no_color or os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def paint(text, code, enabled):
    if not enabled:
        return text
    return f"\033[{code}m{text}\033[0m"


def find_corpus_dirs(selected):
    available = sorted(
        path for path in BASE_DIR.glob(CORPUS_GLOB) if path.is_dir()
    )

    if not selected:
        return available

    by_name = {path.name: path for path in available}
    chosen = []

    for name in selected:
        key = Path(name).name
        if key not in by_name:
            raise SystemExit(
                f"Folder korpus tidak ditemukan: {name}\n"
                f"Tersedia: {', '.join(sorted(by_name))}"
            )
        chosen.append(by_name[key])

    return chosen


def build_pattern(query, is_regex, case_sensitive, loose):
    if is_regex:
        source = query
    else:
        source = re.escape(query)
        if loose:
            source = re.sub(r"(?:\\\s)+", r"\\s+", source)

    flags = re.MULTILINE
    if not case_sensitive:
        flags |= re.IGNORECASE

    try:
        return re.compile(source, flags)
    except re.error as error:
        raise SystemExit(f"Pola regex tidak valid: {error}")


def find_body_start(raw_text):
    if not raw_text.startswith("---\n"):
        return 0

    end = raw_text.find("\n---\n", 4)
    if end == -1:
        return 0

    return end + len("\n---\n")


def read_frontmatter(raw_text):
    if not raw_text.startswith("---\n"):
        return {}

    end = raw_text.find("\n---\n", 4)
    if end == -1:
        return {}

    try:
        return yaml.safe_load(raw_text[4:end]) or {}
    except yaml.YAMLError:
        return {}


def metadata_matches(metadata, filters):
    for key, wanted in filters.items():
        if wanted is None:
            continue
        if str(metadata.get(key, "")).strip().lower() != wanted.strip().lower():
            return False

    return True


def collapse(text):
    return " ".join(text.split())


def build_snippet(raw_text, start, end, context_chars, floor=0):
    left_from = max(floor, start - context_chars)
    right_to = min(len(raw_text), end + context_chars)

    left_raw = raw_text[left_from:start]
    right_raw = raw_text[end:right_to]

    before = collapse(left_raw)
    matched = collapse(raw_text[start:end])
    after = collapse(right_raw)

    if before and left_raw[-1:].isspace():
        before += " "
    if after and right_raw[:1].isspace():
        after = " " + after

    prefix = "..." if left_from > floor else ""
    suffix = "..." if right_to < len(raw_text) else ""

    return prefix + before, matched, after + suffix


def extract_prefilter(query, is_regex, loose):
    """Literal terpanjang dari query, dipakai menyaring file sebelum decode."""
    if is_regex:
        if re.search(r"[|?*+{}\[\]()\\]", query):
            return None

        words = re.findall(r"[0-9A-Za-z]{4,}", query)
        if not words:
            return None
        return max(words, key=len).lower()

    if loose:
        words = re.findall(r"\S+", query)
        if not words:
            return None
        return max(words, key=len).lower()

    return query.lower()


def decode_markdown(raw_bytes):
    """Decode UTF-8 dan samakan dengan perilaku universal newline read_text();
    korpus bercampur CRLF dan LF, sedangkan deteksi frontmatter memakai "\\n"."""
    try:
        raw_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return None

    if "\r" in raw_text:
        raw_text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

    return raw_text


def scan_text(raw_text, path, pattern, options):
    """Cari pattern di satu dokumen yang sudah di-decode. Dipakai bersama oleh
    CLI (baca dari disk) dan dashboard (korpus sudah di-cache di memori)."""
    body_start = 0 if options["include_frontmatter"] else find_body_start(raw_text)

    matches = []
    for match in pattern.finditer(raw_text, body_start):
        if match.start() == match.end():
            continue

        before, matched, after = build_snippet(
            raw_text,
            match.start(),
            match.end(),
            options["context_chars"],
            floor=body_start,
        )
        matches.append(
            {
                "line": raw_text.count("\n", 0, match.start()) + 1,
                "before": before,
                "match": matched,
                "after": after,
            }
        )

        if len(matches) >= options["max_per_file"]:
            break

    if not matches:
        return None

    metadata = read_frontmatter(raw_text)
    if not metadata_matches(metadata, options["filters"]):
        return None

    return {
        "path": str(path),
        "relative_path": str(path.relative_to(BASE_DIR)).replace("\\", "/"),
        "corpus": path.parent.name,
        "chunk_id": metadata.get("chunk_id", path.stem),
        "source_id": metadata.get("source_id", "-"),
        "jenis": metadata.get("jenis", "-"),
        "tipe_dokumen": metadata.get("tipe_dokumen", "-"),
        "dokumen": metadata.get("dokumen", "-"),
        "section": metadata.get("section", "-"),
        "halaman": metadata.get("halaman", "-"),
        "total_matches": len(matches),
        "matches": matches,
    }


def build_options(
    query,
    is_regex=False,
    loose=False,
    case_sensitive=False,
    include_frontmatter=False,
    context_chars=CONTEXT_CHARS,
    max_per_file=MAX_PER_FILE,
    filters=None,
):
    """Siapkan pattern + options sekali, dipakai CLI maupun dashboard."""
    pattern = build_pattern(query, is_regex, case_sensitive, loose)

    prefilter = extract_prefilter(query, is_regex, loose)
    if prefilter and prefilter.isascii():
        prefilter = prefilter.encode("utf-8")
    else:
        prefilter = None

    options = {
        "prefilter": prefilter,
        "include_frontmatter": include_frontmatter,
        "context_chars": max(0, context_chars),
        "max_per_file": max(1, max_per_file),
        "filters": filters or {},
    }

    return pattern, options


def search_file(path, pattern, options):
    try:
        raw_bytes = path.read_bytes()
    except OSError:
        return None

    prefilter = options["prefilter"]
    if prefilter and prefilter not in raw_bytes.lower():
        return None

    raw_text = decode_markdown(raw_bytes)
    if raw_text is None:
        return None

    return scan_text(raw_text, path, pattern, options)


def collect_hits(corpus_dirs, pattern, options):
    hits = []
    scanned = 0

    for corpus_dir in corpus_dirs:
        for path in sorted(corpus_dir.rglob("*.md")):
            scanned += 1
            hit = search_file(path, pattern, options)
            if hit:
                hits.append(hit)

    return hits, scanned


def print_report(hits, scanned, corpus_dirs, args, colored):
    mode = "regex" if args.regex else "literal"
    if args.loose and not args.regex:
        mode += " longgar"
    mode += " case-sensitive" if args.case_sensitive else " case-insensitive"

    print(f'Query : "{args.query}" ({mode})')
    print(f"Korpus: {len(corpus_dirs)} folder, {scanned} file .md dipindai")
    print(f"Hasil : {len(hits)} file cocok")
    print("")

    if not hits:
        print("Tidak ada yang cocok. Coba --loose, kurangi kata, atau pakai --regex.")
        return

    shown = hits[: args.max_results]

    for rank, hit in enumerate(shown, start=1):
        header = f"[{rank}] {hit['relative_path']}"
        print(paint(header, "1;36", colored))
        print(f"    chunk_id : {hit['chunk_id']}")
        print(
            f"    dokumen  : {hit['dokumen']} "
            f"| jenis: {hit['jenis']}/{hit['tipe_dokumen']} "
            f"| halaman: {hit['halaman']}"
        )
        print(f"    section  : {hit['section']}")

        for match in hit["matches"]:
            highlighted = paint(match["match"], "1;33", colored)
            print(
                f"    L{match['line']:<5} {match['before']}"
                f"{highlighted}{match['after']}"
            )

        if hit["total_matches"] >= args.max_per_file:
            print(f"    (dipotong pada {args.max_per_file} match/file)")

        print("")

    if len(hits) > len(shown):
        print(
            f"... {len(hits) - len(shown)} file lain disembunyikan. "
            f"Naikkan dengan --max-results."
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Cari teks literal atau regex di seluruh korpus markdown, "
            "untuk memverifikasi klaim jawaban model RAG."
        )
    )
    parser.add_argument("query", nargs="?", help="Kata/frasa yang dicari")
    parser.add_argument(
        "--regex",
        action="store_true",
        help="Perlakukan query sebagai regex Python",
    )
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Bedakan huruf besar/kecil",
    )
    parser.add_argument(
        "--loose",
        action="store_true",
        help="Spasi pada query cocok dengan spasi/baris baru apa pun",
    )
    parser.add_argument(
        "--corpus",
        action="append",
        help="Batasi ke folder korpus tertentu (boleh diulang)",
    )
    parser.add_argument("--jenis", help="Filter frontmatter: jenis")
    parser.add_argument("--tipe-dokumen", help="Filter frontmatter: tipe_dokumen")
    parser.add_argument("--dokumen", help="Filter frontmatter: dokumen")
    parser.add_argument("--source-id", help="Filter frontmatter: source_id")
    parser.add_argument("--chunk-id", help="Filter frontmatter: chunk_id")
    parser.add_argument(
        "--include-frontmatter",
        action="store_true",
        help="Ikut mencari di dalam blok frontmatter",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=CONTEXT_CHARS,
        help=f"Jumlah karakter konteks di kiri/kanan match (default {CONTEXT_CHARS})",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=MAX_RESULTS,
        help=f"Maksimal file yang ditampilkan (default {MAX_RESULTS})",
    )
    parser.add_argument(
        "--max-per-file",
        type=int,
        default=MAX_PER_FILE,
        help=f"Maksimal match per file (default {MAX_PER_FILE})",
    )
    parser.add_argument(
        "--files-only",
        action="store_true",
        help="Cetak path file saja",
    )
    parser.add_argument("--json", action="store_true", help="Keluarkan JSON")
    parser.add_argument("--no-color", action="store_true", help="Matikan warna")
    parser.add_argument(
        "--list-corpus",
        action="store_true",
        help="Tampilkan daftar folder korpus lalu keluar",
    )
    args = parser.parse_args()

    enable_ansi_on_windows()

    if args.list_corpus:
        for corpus_dir in find_corpus_dirs(None):
            total = sum(1 for _ in corpus_dir.rglob("*.md"))
            print(f"{corpus_dir.name:32} {total} file .md")
        return

    if not args.query:
        raise SystemExit("Query wajib diisi (kecuali memakai --list-corpus).")

    corpus_dirs = find_corpus_dirs(args.corpus)
    if not corpus_dirs:
        raise SystemExit(f"Tidak ada folder {CORPUS_GLOB} di {BASE_DIR}")

    pattern, options = build_options(
        args.query,
        is_regex=args.regex,
        loose=args.loose,
        case_sensitive=args.case_sensitive,
        include_frontmatter=args.include_frontmatter,
        context_chars=args.context,
        max_per_file=args.max_per_file,
        filters={
            "jenis": args.jenis,
            "tipe_dokumen": args.tipe_dokumen,
            "dokumen": args.dokumen,
            "source_id": args.source_id,
            "chunk_id": args.chunk_id,
        },
    )

    hits, scanned = collect_hits(corpus_dirs, pattern, options)

    if args.json:
        print(
            json.dumps(
                {
                    "query": args.query,
                    "regex": args.regex,
                    "scanned_files": scanned,
                    "total_hits": len(hits),
                    "hits": hits[: args.max_results],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.files_only:
        for hit in hits[: args.max_results]:
            print(hit["relative_path"])
        return

    print_report(hits, scanned, corpus_dirs, args, use_color(args.no_color))


if __name__ == "__main__":
    main()
