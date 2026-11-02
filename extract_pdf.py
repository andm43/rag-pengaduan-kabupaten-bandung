"""Ekstraksi teks dari PDF menggunakan GPT-4o vision.

Setiap halaman dirender jadi gambar lalu dikirim ke GPT-4o untuk
transkripsi. Hasilnya: teks terstruktur per halaman yang siap di-chunk
oleh build_corpus_sop.py atau pipeline dashboard.

Dependensi:
    pip install pymupdf openai pillow
"""

import base64
import io
import os
from pathlib import Path

import fitz
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = "gpt-4o"
DPI = 200


SYSTEM_PROMPT = """\
Kamu adalah asisten yang mengekstrak teks dari gambar halaman dokumen PDF.

Instruksi:
1. Transkripsi SELURUH teks yang terlihat di halaman, termasuk header, footer, nomor halaman.
2. Untuk TABEL: konversi ke format teks terstruktur. Gunakan format:
   - Baris header dipisah dengan garis (---).
   - Kolom dipisah dengan tanda pipe (|).
   - Contoh:
     | Kolom 1 | Kolom 2 | Kolom 3 |
     |---------|---------|---------|
     | isi     | isi     | isi     |
3. Untuk FLOWCHART atau DIAGRAM: deskripsikan alurnya secara naratif, langkah per langkah.
   Gunakan penomoran (1, 2, 3, ...) untuk setiap langkah.
   Sertakan kondisi percabangan (jika ada) dengan format: "Jika [kondisi], maka [aksi]."
4. Pertahankan struktur heading/subheading yang terlihat.
5. Jangan menambahkan interpretasi atau komentar — transkripsi saja.
6. Jika halaman kosong atau hanya berisi gambar dekoratif tanpa informasi, tulis: [HALAMAN KOSONG]
7. Gunakan bahasa Indonesia sesuai dokumen sumber.
"""


def render_page_to_base64(page, dpi=DPI):
    """Render satu halaman fitz.Page jadi base64-encoded PNG."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    return base64.b64encode(img_bytes).decode("utf-8")


def extract_page_with_vision(client, img_b64, page_num, total_pages):
    """Kirim gambar halaman ke GPT-4o vision, return teks hasil."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Halaman {page_num} dari {total_pages}. "
                        f"Ekstrak seluruh teks dari halaman ini.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
        max_tokens=4096,
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def extract_pdf(pdf_path, progress_callback=None):
    """Ekstrak seluruh halaman PDF via GPT-4o vision.

    Args:
        pdf_path: path ke file PDF.
        progress_callback(page_num, total, text): dipanggil setelah
            tiap halaman selesai diekstrak.

    Returns:
        list of dict: [{"page": int, "text": str}, ...]
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {pdf_path}")

    client = OpenAI(api_key=OPENAI_API_KEY)
    doc = fitz.open(str(pdf_path))
    total = len(doc)
    results = []

    print(f"[EXTRACT] {pdf_path.name}: {total} halaman")

    for i, page in enumerate(doc):
        page_num = i + 1
        print(f"[PAGE {page_num}/{total}] Rendering + GPT-4o vision...")

        img_b64 = render_page_to_base64(page)
        text = extract_page_with_vision(client, img_b64, page_num, total)

        results.append({"page": page_num, "text": text})
        print(f"[PAGE {page_num}/{total}] OK ({len(text)} chars)")

        if progress_callback:
            progress_callback(page_num, total, text)

    doc.close()
    print(f"[DONE] {total} halaman diekstrak")
    return results


def extract_and_save(pdf_path, output_path=None):
    """Ekstrak PDF dan simpan hasilnya sebagai satu file teks.

    Berguna untuk inspeksi manual sebelum chunking.
    """
    results = extract_pdf(pdf_path)

    if output_path is None:
        output_path = Path(pdf_path).with_suffix(".extracted.md")

    lines = []
    for r in results:
        lines.append(f"# Halaman {r['page']}\n")
        lines.append(r["text"])
        lines.append("\n---\n")

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVED] {output_path}")
    return output_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python extract_pdf.py <file.pdf> [output.md]")
        raise SystemExit(1)

    pdf = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    extract_and_save(pdf, out)
