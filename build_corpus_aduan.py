import pandas as pd
import re
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

from anonymize import anonymize_dataframe, scan_pii


BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "dataset_clean.csv"
INPUT_CSV_NAME = INPUT_CSV.name

OUTPUT_DIR = BASE_DIR / "corpus_aduan"
OUTPUT_DIR.mkdir(exist_ok=True)

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def sanitize_filename(name):

    name = str(name)

    name = re.sub(r'[\\/*?:"<>|]', "_", name)

    return name.strip()


def yaml_quote(value):

    value = str(value).replace("\\", "\\\\").replace('"', '\\"')

    return f'"{value}"'


def clear_output_dir():

    for path in OUTPUT_DIR.glob("*.md"):
        path.unlink()


def _common_parts(row):
    """Enam field yang dipakai bersama oleh dokumen lengkap (build_document)
    dan prefix metadata tiap chunk (build_base_prefix).
    """
    return [
        ("KATEGORI", clean_text(row.get("kategori", ""))),
        ("LOKASI", f"Kecamatan: {clean_text(row.get('kecamatan', ''))}"),
        ("INSTANSI", clean_text(row.get("instansi", ""))),
        ("KLASIFIKASI", clean_text(row.get("klasifikasi", ""))),
        ("STATUS", clean_text(row.get("status", ""))),
        ("JUDUL", clean_text(row.get("judul", ""))),
    ]


def _blocks_to_text(parts):
    return "\n\n".join(f"[{label}]\n{value}" for label, value in parts if value)


def build_document(row):

    parts = _common_parts(row) + [
        ("MASALAH", clean_text(row.get("isi_laporan", ""))),
        ("TINDAK LANJUT", clean_text(row.get("tindak_lanjut", ""))),
    ]

    return _blocks_to_text(parts)


def split_long_text(text, chunk_size=None, chunk_overlap=None):

    effective_chunk_size = chunk_size or CHUNK_SIZE
    effective_chunk_overlap = chunk_overlap or CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=effective_chunk_size,
        chunk_overlap=effective_chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
        keep_separator=False,
    )

    return [
        chunk.strip()
        for chunk in splitter.split_text(clean_text(text))
        if chunk.strip()
    ]


def split_tindak_lanjut_entries(text):

    normalized = clean_text(text)
    if not normalized:
        return []

    marker_pattern = re.compile(
        r"\[?\s*TINDAK\s+LANJUT\s+(\d+)\s*\]?", re.IGNORECASE
    )
    matches = list(marker_pattern.finditer(normalized))

    if not matches:
        return [
            {
                "label": "TINDAK LANJUT",
                "content": normalized,
            }
        ]

    entries = []

    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(normalized)
        content = normalized[start:end].strip(" :-[]")
        if not content:
            continue

        entries.append(
            {
                "label": f"TINDAK LANJUT {match.group(1)}",
                "content": content,
            }
        )

    return entries


def build_base_prefix(row):
    return _blocks_to_text(_common_parts(row)).strip()


def build_section_chunk(section_name, content, prefix=""):
    chunk_parts = []

    if prefix:
        chunk_parts.append(prefix.strip())

    chunk_parts.append(f"[{section_name}]\n{content.strip()}")
    return "\n\n".join(part for part in chunk_parts if part).strip()


def split_document_from_row(row):
    full_document = build_document(row)
    chunks = []

    if full_document:
        chunks.append(
            {
                "section": "Dokumen Lengkap",
                "content": full_document,
            }
        )

    prefix = build_base_prefix(row)
    masalah = clean_text(row.get("isi_laporan", ""))
    tindak_lanjut = clean_text(row.get("tindak_lanjut", ""))

    if masalah:
        masalah_chunks = split_long_text(
            masalah,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
    else:
        masalah_chunks = []

    if tindak_lanjut:
        tindak_entries = split_tindak_lanjut_entries(tindak_lanjut)
    else:
        tindak_entries = []

    if masalah_chunks:
        chunks.append(
            {
                "section": "Ringkasan Metadata dan Masalah",
                "content": build_section_chunk(
                    "MASALAH",
                    masalah_chunks[0],
                    prefix=prefix,
                ),
            }
        )

        for extra_masalah in masalah_chunks[1:]:
            chunks.append(
                {
                    "section": "Masalah",
                    "content": build_section_chunk("MASALAH", extra_masalah),
                }
            )
    elif prefix:
        chunks.append(
            {
                "section": "Metadata Dasar",
                "content": prefix,
            }
        )

    if tindak_entries:
        groups = []
        current = []
        current_len = 0

        for entry in tindak_entries:
            block = build_section_chunk(entry["label"], entry["content"])

            if len(block) > CHUNK_SIZE:
                if current:
                    groups.append(current)
                    current, current_len = [], 0

                parts = split_long_text(
                    entry["content"],
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                )
                for part_index, part in enumerate(parts, start=1):
                    groups.append(
                        [
                            {
                                "label": f"{entry['label']} Bagian {part_index}",
                                "content": part,
                            }
                        ]
                    )
                continue

            if current and current_len + len(block) > CHUNK_SIZE:
                groups.append(current)
                current, current_len = [], 0

            current.append(entry)
            current_len += len(block)

        if current:
            groups.append(current)

        for group in groups:
            if len(group) == 1:
                section = group[0]["label"].title()
            else:
                section = f"{group[0]['label'].title()} - {group[-1]['label'].title()}"

            chunks.append(
                {
                    "section": section,
                    "content": "\n\n".join(
                        build_section_chunk(item["label"], item["content"])
                        for item in group
                    ),
                }
            )

    unique_chunks = []
    seen = set()
    for chunk in chunks:
        normalized = chunk["content"].strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_chunks.append(
            {
                "section": chunk["section"],
                "content": normalized,
            }
        )

    return unique_chunks


def build_metadata(row, source_id, chunk_id, chunk_index, chunk_total, section_name):

    metadata = {
        "source_id": source_id,
        "chunk_id": chunk_id,
        "id": source_id,
        "jenis": "pengaduan",
        "tipe_dokumen": "data_pengaduan",
        "dokumen": "Data Pengaduan Masyarakat Kabupaten Bandung",
        "sumber_file": INPUT_CSV_NAME,
        "section": section_name,
        "tahun": clean_text(row.get("tahun", "")),
        "chunk_index": str(chunk_index),
        "chunk_total": str(chunk_total),
        "kategori": clean_text(row.get("kategori", "")),
        "kecamatan": clean_text(row.get("kecamatan", "")),
        "instansi": clean_text(row.get("instansi", "")),
        "klasifikasi": clean_text(row.get("klasifikasi", "")),
        "status": clean_text(row.get("status", "")),
        "judul": clean_text(row.get("judul", "")),
        "tanggal_masuk": clean_text(row.get("tanggal_masuk", "")),
        "tanggal_tindak_lanjut_pertama": clean_text(row.get("tanggal_tindak_lanjut_pertama", "")),
        "tanggal_selesai": clean_text(row.get("tanggal_selesai", "")),
        "durasi_respon": clean_text(row.get("durasi_respon", "")),
        "durasi_penyelesaian": clean_text(row.get("durasi_penyelesaian", "")),
    }

    yaml_lines = []

    for k, v in metadata.items():

        if v:
            yaml_lines.append(f"{k}: {yaml_quote(v)}")

    return "\n".join(yaml_lines)


def main():

    print(f"[LOAD] {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    print(f"[INFO] Total data: {len(df)}")

    df = anonymize_dataframe(df)

    print("[INFO] Anonimisasi diterapkan pada judul, isi_laporan, tindak_lanjut")

    print("[BUILD] Generate markdown corpus...")

    pending_files = []
    leaks = []

    for idx, row in df.iterrows():

        doc_id = clean_text(row.get("id", ""))

        if not doc_id:
            doc_id = f"DOC_{idx}"

        source_id = doc_id
        chunks = split_document_from_row(row)

        if not chunks:
            continue

        chunk_total = len(chunks)

        for chunk_index, chunk in enumerate(chunks):

            chunk_id = f"pengaduan-{source_id}-{chunk_index:02d}"
            filename = sanitize_filename(chunk_id) + ".md"
            filepath = OUTPUT_DIR / filename

            metadata = build_metadata(
                row,
                source_id,
                chunk_id,
                chunk_index,
                chunk_total,
                chunk["section"],
            )

            markdown = f"""---
{metadata}
---

{chunk["content"]}
"""

            for jenis, cuplikan in scan_pii(markdown):
                leaks.append((chunk_id, jenis, cuplikan))

            pending_files.append((filepath, markdown))

    if leaks:
        print(f"\n[GAGAL] Terdeteksi {len(leaks)} sisa identitas pada korpus.")
        print("[GAGAL] Tidak ada berkas ditulis. corpus_md/ dibiarkan apa adanya.\n")

        for chunk_id, jenis, cuplikan in leaks[:20]:
            print(f"  - {chunk_id} [{jenis}] {cuplikan!r}")

        if len(leaks) > 20:
            print(f"  ... dan {len(leaks) - 20} lainnya")

        raise SystemExit(
            "Korpus tidak dibangun karena masih mengandung identitas. "
            "Perbaiki pola di anonymize.py lalu jalankan ulang."
        )

    print("[OK] Pemindaian PII bersih, tidak ada sisa identitas.")

    clear_output_dir()

    for filepath, markdown in pending_files:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(markdown)

    print(f"[DONE] Saved to: {OUTPUT_DIR}")
    print(f"[INFO] Total chunks saved: {len(pending_files)}")


if __name__ == "__main__":
    main()
