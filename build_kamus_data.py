"""Bangun chunk "kamus data" dari metadata korpus pengaduan.

LATAR BELAKANG
--------------
Metadata kategori/klasifikasi/instansi/kecamatan/status melekat pada tiap
laporan satu per satu (lihat build_corpus_aduan.py). Tidak ada satu pun chunk
di korpus yang memuat DAFTAR LENGKAP nilai-nilai itu. Akibatnya pertanyaan
agregat seperti "klasifikasi ada apa saja?" atau "ada berapa kategori aduan?"
tidak punya apa pun untuk di-retrieve -- sistem hanya bisa menjawab dari satu
aduan acak.

Skrip ini menyusun daftar tersebut sebagai chunk korpus biasa. Seluruh isinya
DITURUNKAN dari metadata korpus, tidak ada nilai yang ditulis tangan.

Posisi dalam pipeline:
    dataset_clean.csv
        -> build_corpus_aduan.py     -> corpus_aduan/pengaduan-*.md
        -> build_kamus_data.py       -> corpus_aduan/kamus-*.md      (skrip ini)
        -> build_chroma_index.py     -> chroma_rag/
        -> build_bm25_index.py       -> chroma_rag/bm25_index.pkl

CATATAN PENTING soal cara menghitung
------------------------------------
Jumlah dihitung per LAPORAN UNIK (source_id), bukan per chunk. Satu laporan
bisa terpecah menjadi beberapa chunk, sehingga menghitung per chunk akan
menggelembungkan angka. Hasil per-laporan cocok dengan rekap sumber:
Pengaduan Berkadar Pengawasan 39,76%, Tidak Berkadar Pengawasan 30,73%,
Permintaan Informasi 5,81%, Aspirasi 2,80%.
"""

import collections
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parent
CORPUS_DIR = BASE_DIR / "corpus_aduan"

FACETS = [
    ("klasifikasi", "kamus-klasifikasi-laporan", "Daftar Klasifikasi Laporan", None),
    ("status", "kamus-status-laporan", "Daftar Status Penanganan Laporan", None),
    ("kecamatan", "kamus-kecamatan", "Daftar Kecamatan pada Data Pengaduan", None),
    ("instansi", "kamus-instansi", "Daftar Instansi Tujuan Pengaduan", 45),
    ("kategori", "kamus-kategori", "Daftar Kategori Masalah Pengaduan", 55),
]


def read_report_metadata():
    """Baca frontmatter tiap chunk, ambil satu metadata per LAPORAN unik."""
    by_report = {}
    for path in sorted(CORPUS_DIR.glob("*.md")):
        if path.name.startswith("kamus-"):
            continue
        raw = path.read_text(encoding="utf-8")
        if not raw.startswith("---\n"):
            continue
        parts = raw.split("---\n", 2)
        if len(parts) < 3:
            continue
        meta = yaml.safe_load(parts[1]) or {}
        source_id = meta.get("source_id")
        if source_id and source_id not in by_report:
            by_report[source_id] = meta
    return by_report


def tally(reports, field):
    return collections.Counter(
        str(m.get(field)).strip()
        for m in reports.values()
        if str(m.get(field) or "").strip()
    ).most_common()


def frontmatter(chunk_id, section, source_id):
    return (
        "---\n"
        f'source_id: "{source_id}"\n'
        f'chunk_id: "{chunk_id}"\n'
        f'id: "{source_id}"\n'
        'jenis: "pengaduan"\n'
        'tipe_dokumen: "kamus_data"\n'
        'dokumen: "Kamus Data Pengaduan Masyarakat Kabupaten Bandung"\n'
        'sumber_file: "dataset_clean.csv"\n'
        f'section: "{section}"\n'
        "---\n\n"
    )


def body_klasifikasi(values, total):
    lines = [f"- {v}: {n} laporan ({100 * n / total:.2f}%)" for v, n in values]
    return (
        f"Field KLASIFIKASI pada data pengaduan Kabupaten Bandung memiliki "
        f"{len(values)} nilai yang sah, dihitung dari {total} laporan:\n"
        + "\n".join(lines)
        + "\n\n"
        "Catatan penting: KLASIFIKASI tidak sama dengan JENIS PENGADUAN. "
        "Menurut Peraturan Bupati Bandung Nomor 290 Tahun 2023 Pasal 13, Jenis "
        "Pengaduan hanya terdiri dari dua, yaitu Pengaduan Tidak Berkadar Pengawasan "
        "dan Pengaduan Berkadar Pengawasan. Adapun Permintaan Informasi dan Aspirasi "
        "bukan jenis pengaduan, melainkan jenis laporan lain yang juga ditampung "
        "SP4N-LAPOR. Nilai BELUM DIKLASIFIKASI berarti laporan belum diklasifikasikan "
        "oleh admin."
    )


def body_status(values, total):
    lines = [f"- {v}: {n} laporan ({100 * n / total:.2f}%)" for v, n in values]
    return (
        f"Field STATUS pada data pengaduan Kabupaten Bandung memiliki "
        f"{len(values)} nilai, dihitung dari {total} laporan:\n" + "\n".join(lines)
    )


def body_kecamatan(values, total):
    lines = [f"- {v}: {n}" for v, n in values]
    return (
        f"Daftar kecamatan yang tercatat pada data pengaduan Kabupaten Bandung, "
        f"{len(values)} nilai dari {total} laporan (TIDAK DIKETAHUI berarti "
        f"pelapor tidak mencantumkan lokasi):\n" + "\n".join(lines)
    )


def body_instansi(part, index, total_values, total):
    lines = [f"- {v}: {n}" for v, n in part]
    return (
        f"Daftar instansi tujuan pada data pengaduan Kabupaten Bandung "
        f"(bagian {index}, total {total_values} instansi dari {total} laporan), "
        f"diurutkan dari yang paling banyak menerima laporan:\n" + "\n".join(lines)
    )


def body_kategori(part, index, total_values, total):
    lines = [f"- {v}: {n}" for v, n in part]
    return (
        f"Daftar kategori masalah pada data pengaduan Kabupaten Bandung "
        f"(bagian {index}, total {total_values} kategori dari {total} laporan), "
        f"diurutkan dari yang paling sering diadukan:\n" + "\n".join(lines)
    )


def write_chunk(chunk_id, section, source_id, body):
    path = CORPUS_DIR / f"{chunk_id}.md"
    path.write_text(
        frontmatter(chunk_id, section, source_id) + body.strip() + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def main():
    reports = read_report_metadata()
    total = len(reports)
    print(f"[LOAD] laporan unik: {total}")

    written = []
    for field, base_id, section, per_chunk in FACETS:
        values = tally(reports, field)
        if not values:
            print(f"[SKIP] {field}: tidak ada nilai")
            continue

        if per_chunk is None:
            if field == "klasifikasi":
                body = body_klasifikasi(values, total)
            elif field == "status":
                body = body_status(values, total)
            else:
                body = body_kecamatan(values, total)
            written.append(write_chunk(base_id, section, f"kamus-{field}", body))
        else:
            for start in range(0, len(values), per_chunk):
                part = values[start:start + per_chunk]
                index = start // per_chunk + 1
                if field == "instansi":
                    body = body_instansi(part, index, len(values), total)
                else:
                    body = body_kategori(part, index, len(values), total)
                written.append(
                    write_chunk(
                        f"{base_id}-{index:02d}",
                        f"{section} (bagian {index})",
                        f"kamus-{field}",
                        body,
                    )
                )
        print(f"[OK]   {field}: {len(values)} nilai")

    print(f"\n[DONE] {len(written)} chunk kamus ditulis ke {CORPUS_DIR}")


if __name__ == "__main__":
    main()
