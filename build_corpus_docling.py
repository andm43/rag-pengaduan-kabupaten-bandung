import argparse
import html
import re
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "pendukung"

OUTPUT_DIR = BASE_DIR / "corpus_regulasi_sop"
OUTPUT_DIR.mkdir(exist_ok=True)

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
IMAGES_SCALE = 2.0

PAGE_OPEN = "\ue000"
PAGE_CLOSE = "\ue001"
PAGE_RE = re.compile(r"\ue000(\d+)\ue001")


def strip_page_marks(text):
    return PAGE_RE.sub("", text)


VISION_PROVIDER = "gpt4o"
VISION_MODEL_CLAUDE = "claude-sonnet-4-6"
VISION_MODEL_GPT4O = "gpt-4o"
VISION_RENDER_DPI = 200

_LEGAL_CONTEXT = """Dokumen ini adalah produk hukum resmi Pemerintah Kabupaten Bandung,
Indonesia (Peraturan Bupati atau Keputusan Bupati) termasuk kategori
peraturan perundang-undangan. Berdasarkan Pasal 42 Undang-Undang Nomor 28
Tahun 2014 tentang Hak Cipta, peraturan perundang-undangan Indonesia TIDAK
memiliki perlindungan hak cipta dan bersifat terbuka untuk publik.
Transkripsi ini dibutuhkan untuk membangun basis pengetahuan sistem
informasi publik pelayanan pengaduan masyarakat (bukan untuk tujuan
komersial atau pelanggaran hak apapun)."""

VISION_EXTRACTION_PROMPT = f"""{_LEGAL_CONTEXT}

Ini adalah 1 halaman dari dokumen tersebut (dokumen digital asli, bukan
hasil scan). Halaman ini terdeteksi mengandung TABEL dan/atau GAMBAR/DIAGRAM
(termasuk kemungkinan form/template kosong, atau bagan alur/flowchart).

TUGAS: transkripsikan SELURUH isi halaman ini ke Markdown, selengkap dan
sepersis mungkin dengan struktur asli.

ATURAN:
1. Kalau ada TABEL tulis sebagai tabel Markdown standar (| kolom |
   kolom |), pertahankan SEMUA kolom dan baris, termasuk header bertingkat
   (kalau ada 2 baris header yang bertingkat/merged cell, tulis header
   gabungan yang jelas). Kalau tabelnya FORM KOSONG (baris data tidak
   terisi), tetap tulis strukturnya, jangan mengarang isi data yang tidak ada.
2. Kalau ada BAGAN ALUR/FLOWCHART/DIAGRAM -- tulis sebagai narasi langkah
   bernomor yang menjelaskan alur (kotak apa terhubung ke kotak apa, label
   panahnya apa), JANGAN dipaksa jadi tabel.
3. Kalau ada TEKS PASAL/PARAGRAF biasa di halaman yang sama -- tulis
   sebagai teks Markdown biasa, pertahankan struktur (BAB, Pasal, ayat,
   penomoran) PERSIS seperti aslinya, jangan diringkas atau dihilangkan.
4. Kalau ada kode/angka (misal kode klasifikasi seperti "0100", "0101",
   nomor pasal/ayat), PASTIKAN semua angka tersebut ikut tertulis persis,
   jangan sampai hilang atau salah ketik -- ini bagian paling penting dan
   paling sering gagal di ekstraksi OCR biasa.
5. Kalau ada bagian yang tidak terbaca jelas / meragukan, tulis eksplisit
   "[TIDAK YAKIN: ...]" -- JANGAN menebak dengan percaya diri seolah pasti benar.
6. Beri judul singkat (## Judul) di awal.

Output HANYA Markdown, tanpa komentar/penjelasan tambahan di luar itu."""


_VISION_REFUSAL_KEYWORDS = [
    "i'm sorry", "i am sorry", "i can't assist", "i cannot assist",
    "i'm unable to", "i am unable to", "can't help with that",
    "cannot help with that", "unable to transcribe",
    "unable to provide a transcription", "unable to display",
    "i can't help", "i can't reproduce", "i cannot reproduce",
    "saya maaf", "maaf, saya", "mohon maaf", "saya tidak dapat",
    "saya tidak bisa", "tidak dapat membantu", "tidak bisa membantu",
    "tidak dapat memproses", "copyrighted",
]

_VISION_HALLUCINATION_MARKERS = [
    "berdasarkan deskripsi", "silakan lengkapi", "contoh isi/deskripsi",
    "contoh isi", "contoh deskripsi", "based on your description",
    "please provide", "as an example", "berikut adalah format",
    "berikut adalah contoh", "namun, berdasarkan",
]


def _is_vision_refusal(text):
    t = text.strip()
    t_lower = t.lower()
    if len(t) <= 300 and any(kw in t_lower for kw in _VISION_REFUSAL_KEYWORDS):
        return True
    if any(kw in t_lower[:150] for kw in _VISION_REFUSAL_KEYWORDS):
        return True
    if any(kw in t_lower for kw in _VISION_HALLUCINATION_MARKERS):
        return True
    return False


PICTURE_MIN_AREA_RATIO = 0.08


def _significant_pictures(doc, page_no):
    """Saring document.pictures: buang yang keliatan kayak logo/kop surat
    kecil (luas < PICTURE_MIN_AREA_RATIO dari luas halaman). Kalau atribut
    bbox/size tidak tersedia (versi docling beda / gagal baca), gambar
    itu TETAP dianggap signifikan -- fail-safe ke arah "tetap coba vision"
    daripada diam-diam melewatkan flowchart asli."""
    page_size = None
    try:
        page_size = doc.pages[page_no].size
    except Exception:
        pass

    result = []
    for pic in doc.pictures:
        try:
            bbox = pic.prov[0].bbox
            pic_area = abs(bbox.r - bbox.l) * abs(bbox.t - bbox.b)
            page_area = page_size.width * page_size.height
            ratio = pic_area / page_area if page_area > 0 else 1.0
            if ratio >= PICTURE_MIN_AREA_RATIO:
                result.append(pic)
        except Exception:
            result.append(pic)
    return result


def _load_dotenv_once():
    import os
    for p in [BASE_DIR / ".env", Path.cwd() / ".env"]:
        if p.exists():
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    k, v = k.strip(), v.strip().strip('"').strip("'")
                    if k and k not in os.environ:
                        os.environ[k] = v
            return


_load_dotenv_once()


def _render_page_png(pdf_path, page_no):
    import fitz
    with fitz.open(str(pdf_path)) as doc:
        pix = doc[page_no - 1].get_pixmap(dpi=VISION_RENDER_DPI)
        return pix.tobytes("png")


def _call_claude_vision(image_bytes, api_key):
    import base64
    import json
    from urllib.request import urlopen, Request

    b64 = base64.b64encode(image_bytes).decode("ascii")
    payload = {
        "model": VISION_MODEL_CLAUDE,
        "max_tokens": 2500,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                {"type": "text", "text": VISION_EXTRACTION_PROMPT},
            ],
        }],
    }
    req = Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
    return result["content"][0]["text"].strip()


def _call_gpt4o_vision(image_bytes, api_key):
    import base64
    import json
    from urllib.request import urlopen, Request

    b64 = base64.b64encode(image_bytes).decode("ascii")
    payload = {
        "model": VISION_MODEL_GPT4O,
        "max_tokens": 2500,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": VISION_EXTRACTION_PROMPT},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{b64}",
                    "detail": "high",
                }},
            ],
        }],
    }
    req = Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
    return result["choices"][0]["message"]["content"].strip()


def _vision_extract_page(pdf_path, page_no):
    """Render 1 halaman jadi PNG, kirim ke provider vision aktif
    (VISION_PROVIDER: "claude" atau "gpt4o"), kembalikan teks hasil
    transkripsi. Return None (sinyal "pakai fallback docling") kalau
    API key tidak ada, request gagal, atau modelnya menolak."""
    import os

    if VISION_PROVIDER == "claude":
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        env_name = "ANTHROPIC_API_KEY"
        call_fn = _call_claude_vision
    elif VISION_PROVIDER == "gpt4o":
        api_key = os.getenv("OPENAI_API_KEY", "")
        env_name = "OPENAI_API_KEY"
        call_fn = _call_gpt4o_vision
    else:
        raise ValueError(f"VISION_PROVIDER tidak dikenal: {VISION_PROVIDER!r}")

    if not api_key:
        print(f"    [WARN] {env_name} tidak ada (.env) -- lewati vision, "
              f"pakai teks docling untuk halaman ini.")
        return None

    image_bytes = _render_page_png(pdf_path, page_no)

    try:
        raw = call_fn(image_bytes, api_key)
        raw = re.sub(r"^```(?:markdown)?\s*", "", raw)
        raw = re.sub(r"```\s*$", "", raw)
        raw = raw.strip()
    except Exception as exc:
        print(f"    [WARN] Vision API ({VISION_PROVIDER}) gagal ({exc}) -- "
              f"pakai teks docling untuk halaman ini.")
        return None

    if _is_vision_refusal(raw):
        print(f"    [WARN] Vision API ({VISION_PROVIDER}) menolak halaman ini "
              f"({raw[:60]!r}...) -- pakai teks docling untuk halaman ini.")
        return None

    if "[TIDAK YAKIN" in raw:
        print("    [INFO] Ada bagian ditandai '[TIDAK YAKIN...]' oleh vision -- "
              "perlu cek manual.")

    return raw


def _rotation_ocr_score(img_path):
    import csv
    import io
    import statistics
    import subprocess

    try:
        out = subprocess.run(
            [TESSERACT_CMD, str(img_path), "-", "--psm", "6", "-l", "ind", "tsv"],
            capture_output=True, text=True, timeout=180,
            encoding="utf-8", errors="replace",
        ).stdout
    except Exception:
        return 0.0

    confs = []
    for row in csv.DictReader(io.StringIO(out), delimiter="\t", quoting=csv.QUOTE_NONE):
        try:
            c = float(row["conf"])
        except (TypeError, ValueError, KeyError):
            continue
        if c >= 0 and len(row.get("text", "").strip()) > 2:
            confs.append(c)

    if not confs:
        return 0.0
    return statistics.mean(confs) * (len(confs) ** 0.5)


def _detect_page_rotation(pdf_path, page_no, workdir):
    import re as _re
    import subprocess

    import fitz

    osd_img = workdir / f"osd_{page_no}.png"
    with fitz.open(str(pdf_path)) as doc:
        doc[page_no - 1].get_pixmap(dpi=200).save(str(osd_img))

    try:
        out = subprocess.run(
            [TESSERACT_CMD, str(osd_img), "-", "--psm", "0"],
            capture_output=True, text=True, timeout=120,
            encoding="utf-8", errors="replace",
        ).stdout
    except Exception:
        out = ""

    m = _re.search(r"^Rotate:\s*(\d+)", out, _re.MULTILINE)
    candidate = int(m.group(1)) % 360 if m else 0
    if candidate == 0:
        return 0

    alternatives = sorted({0, candidate, (candidate + 180) % 360})
    scored = []
    for deg in alternatives:
        img = workdir / f"cand_{page_no}_{deg}.png"
        with fitz.open(str(pdf_path)) as doc:
            page = doc[page_no - 1]
            page.set_rotation((page.rotation + deg) % 360)
            page.get_pixmap(dpi=150).save(str(img))
        scored.append((_rotation_ocr_score(img), deg))

    scored.sort(key=lambda item: (-item[0], item[1]))
    best_score, best = scored[0]
    return best if best_score > 0 else 0


def _build_deskewed_pdf(pdf_path, total_pages, workdir):
    """Salinan pdf_path dengan tiap halaman yang terdeteksi miring
    diputar ke posisi tegak. Dipakai sebagai sumber untuk docling MAUPUN
    render vision, supaya keduanya membaca dari halaman yang sama-sama
    sudah tegak (bukan cuma salah satunya)."""
    import fitz

    out_path = workdir / f"{sanitize_filename(pdf_path.stem)}_deskewed.pdf"
    doc = fitz.open(str(pdf_path))
    rotations = {}
    for page_no in range(1, total_pages + 1):
        deg = _detect_page_rotation(pdf_path, page_no, workdir)
        rotations[page_no] = deg
        if deg:
            page = doc[page_no - 1]
            page.set_rotation((page.rotation + deg) % 360)
    doc.save(str(out_path))
    doc.close()
    return out_path, rotations


def _has_digital_text_layer(pdf_path, page_no, min_chars=20):
    """True kalau halaman ini punya layer teks digital ASLI (bukan hasil
    OCR) -- dicek dari fitz page.get_text() SEBELUM docling/OCR ikut
    campur. Dipakai sebagai sinyal TAMBAHAN untuk memicu vision, karena
    terbukti (SOP - SOP Pengaduan.pdf) docling.document.tables/pictures
    TIDAK terisi sama sekali untuk halaman yang seluruhnya berupa satu
    gambar scan penuh -- dari sudut pandang docling tidak ada "gambar DI
    DALAM halaman" untuk dideteksi kalau seluruh halamannya sendiri
    adalah gambar. Sinyal document.tables/pictures yang sudah terbukti
    andal untuk PDF born-digital (Lampiran II/IV No-38-Tahun-2005.pdf)
    tetap dipakai; ini cuma jaring pengaman TAMBAHAN untuk kasus yang
    tidak tercakup sinyal itu."""
    import fitz
    with fitz.open(str(pdf_path)) as doc:
        text = doc[page_no - 1].get_text()
    return len(text.strip()) >= min_chars


def build_converter():
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.images_scale = IMAGES_SCALE
    pipeline_options.ocr_options = TesseractCliOcrOptions(
        lang=["ind"],
        tesseract_cmd=TESSERACT_CMD,
        force_full_page_ocr=True,
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def extract_pdf(converter, pdf_path):
    """Ekstraksi teks PDF per halaman dengan Docling.

    Setiap halaman dikonversi terpisah (page_range=(page_no, page_no)) dan diberi
    penanda PAGE_OPEN/PAGE_CLOSE agar nomor halaman tiap section dapat dilacak.
    Konversi per halaman juga membatasi pemakaian memori OCR, sehingga kegagalan
    satu halaman tidak mengosongkan sisa dokumen.

    Teks Docling selalu dihitung sebagai baseline. Ekstraksi vision dicoba bila
    Docling mendeteksi tabel atau gambar pada halaman, atau bila halaman tidak
    memiliki lapisan teks digital (_has_digital_text_layer()). Hasil vision hanya
    dipakai bila berhasil, selain itu teks Docling dipertahankan.

    Sebelum dibaca, rotasi tiap halaman diperiksa dari isi visualnya
    (_detect_page_rotation) dan diputar tegak bila perlu.
    """
    total_pages = _count_pages(pdf_path)

    import tempfile
    with tempfile.TemporaryDirectory() as workdir_str:
        workdir = Path(workdir_str)
        source_path, rotations = _build_deskewed_pdf(pdf_path, total_pages, workdir)
        fixed = {k: v for k, v in rotations.items() if v}
        if fixed:
            print(f"  [ROTASI] {pdf_path.name}: " +
                  ", ".join(f"hal {k}->{v}°" for k, v in fixed.items()))

        pages_text = []
        for page_no in range(1, total_pages + 1):
            doc = None
            try:
                result = converter.convert(str(source_path), page_range=(page_no, page_no))
                doc = result.document
                page_md = doc.export_to_markdown(page_no=page_no)
            except Exception as exc:
                print(f"[WARN] Gagal OCR halaman {page_no} ({pdf_path.name}): {exc}")
                page_md = ""

            plain_text = strip_markdown_syntax(page_md)
            plain_text = fix_spaced_text(plain_text)

            pictures_here = _significant_pictures(doc, page_no) if doc is not None else []
            has_table_or_picture = doc is not None and (len(doc.tables) > 0 or len(pictures_here) > 0)
            is_scan_page = not _has_digital_text_layer(source_path, page_no)

            if VISION_PROVIDER != "none" and (has_table_or_picture or is_scan_page):
                vision_text = _vision_extract_page(source_path, page_no)
                if vision_text:
                    plain_text = strip_markdown_syntax(vision_text)
                    plain_text = fix_spaced_text(plain_text)

            pages_text.append(f"{PAGE_OPEN}{page_no}{PAGE_CLOSE}{plain_text}")

        return "\n".join(pages_text)


def _count_pages(pdf_path):
    """
    Hitung jumlah halaman lewat PyMuPDF (murah, tanpa OCR) -- dipakai
    supaya extract_pdf() tidak perlu convert() seluruh dokumen sekaligus
    cuma untuk tahu total halaman.
    """
    import fitz
    with fitz.open(str(pdf_path)) as doc:
        return doc.page_count


def _table_row_text(cells):
    """
    Render satu baris tabel markdown docling jadi teks.

    TableFormer kadang salah mengenali blok enumerasi preambul
    (Menimbang/Mengingat, yang di PDF aslinya berjajar rapi seperti
    kolom) sebagai TABEL DATA, menghasilkan baris seperti
    "Menimbang | : | a. | bahwa ...". Ini bukan tabel data asli --
    kalau dipertahankan dengan "|", hasilnya prosa yang penuh garis
    pemisah yang membingungkan.

    Ciri pembeda: tabel data asli (mis. tabel klasifikasi LAMPIRAN)
    biasanya punya sel panjang (isi/deskripsi) di TENGAH, diapit sel
    pendek (nomor/kode) di kedua sisi. Baris enumerasi palsu punya sel
    panjang (isi kalimat) di sel TERAKHIR, didahului sel-sel pendek
    (label "Menimbang", tanda baca ":", penomoran "a."). Kalau pola itu
    terdeteksi, baris dibaca sebagai prosa biasa (digabung spasi, tanpa
    "|"); selain itu dipertahankan sebagai baris tabel kolom.
    """
    long_indices = [i for i, c in enumerate(cells) if len(c) > 15]
    if len(long_indices) == 1 and long_indices[0] == len(cells) - 1:
        return " ".join(cells)
    return " | ".join(cells)


def strip_markdown_syntax(text):
    """
    Normalisasi markdown docling (#, -, *, |) jadi teks polos.

    WAJIB dijalankan sebelum clean_text/split_sections/extract_title --
    fungsi-fungsi itu disalin persis dari pipeline PyMuPDF lama dan
    mengasumsikan teks polos per baris. Tanpa ini, tanda "## " dan "- "
    ikut kebawa jadi judul section/nama file, dan baris-baris list
    markdown malah ikut tergabung jadi satu paragraf oleh clean_text().
    """
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.replace("&#124;", "/")
        stripped = line.strip()

        if stripped and re.match(r"^[\s|:\-]+$", line):
            continue
        if re.match(r"^\s*<!--\s*image\s*-->\s*$", line, re.IGNORECASE):
            continue
        line = re.sub(r"^\s{0,3}#{1,6}\s*", "", line)
        line = re.sub(r"^\s*[-*+]\s+", "", line)
        line = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
        line = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", line)

        if re.match(r"^\|.*\|$", stripped):
            cells = [c.strip() for c in stripped.strip("|").split("|") if c.strip()]
            if cells:
                cleaned.append("")
                cleaned.append(_table_row_text(cells))
                cleaned.append("")
            continue

        line = line.replace("|", " ")
        cleaned.append(line)
    return "\n".join(cleaned)


HEADER_LABELS = ("NOMOR", "LAMPIRAN", "TANGGAL", "SERI")

ORPHAN_LABEL_RE = re.compile(
    r"^(?:" + "|".join(HEADER_LABELS) + r")\s*:\s*$",
    re.IGNORECASE,
)

BLOCK_HEADING_RE = re.compile(
    r"^(?:TENTANG|MEMUTUSKAN|MENETAPKAN|Menimbang|Mengingat|BUPATI\b)",
    re.IGNORECASE,
)


def strip_leaked_page_numbers(text):
    """
    Nomor halaman yang bocor ke badan teks. Di KEPUTUSAN Bupati, kepala
    halaman 2 tertulis "- 2 -"; OCR docling kehilangan tanda hubungnya
    sehingga tersisa "2." yang menempel di depan butir dasar hukum:

        ... Nomor 6856):
        2. 4 Peraturan Pemerintah Nomor 96 tahun 2012 ...

    Angka "4" itu penanda butir yang ASLI (di naskahnya memang ditulis tanpa
    titik), sedangkan "2." murni sampah. Pembuang nomor halaman di atas tidak
    menangkapnya karena menuntut pola "- N -" utuh satu baris.

    DUA syarat harus terpenuhi sekaligus sebelum angka dibuang, dan keduanya
    perlu -- syarat nilai saja jelas tidak cukup:

    1. Nilainya sama dengan nomor halaman tempat dia berada (diambil dari
       marker, bukan ditebak dari teks).
    2. Langsung diikuti angka lain lalu huruf kapital, yaitu penanda butir
       asli yang tertimpa.

    Diukur ke seluruh korpus: pola dua-angka ini cuma muncul 2 kali, dan
    hanya 1 yang lolos syarat nilai (kasus KEPUTUSAN di atas). Satunya lagi
    "12. 4 Fungsi Pengelola Pengaduan ..." di Perbup 290/2023 tersaring benar
    karena 12 bukan nomor halamannya. Sengaja sesempit ini: aturan yang lebih
    longgar (buang SEMUA angka telanjang yang nilainya sama dengan nomor
    halaman) sempat diuji dan ternyata ikut menghapus penanda butir asli
    seperti "2. Pemerintah Daerah adalah ..." di Pasal 1.
    """
    potongan = PAGE_RE.split(text)

    hasil = [potongan[0]]
    for i in range(1, len(potongan), 2):
        nomor, isi = potongan[i], potongan[i + 1]
        isi = re.sub(
            r"(?<![\d.,/-])\b%s\.\s+(?=\d{1,2}\s+[A-Z])" % re.escape(nomor),
            "",
            isi,
        )
        hasil.append(f"{PAGE_OPEN}{nomor}{PAGE_CLOSE}{isi}")

    return "".join(hasil)


def fix_header_labels(text):
    """
    Blok kepala dokumen (NOMOR/LAMPIRAN/TANGGAL) di keputusan & peraturan
    disusun dua kolom: label di kiri, nilai di kanan. Docling membaca tiap
    sel sebagai baris terpisah, jadi labelnya jadi yatim:

        NOMOR :
        000.8.1.1/KEP.555-DISKOMINFO/2024 2
        LAMPIRAN :
        (DUA)

    Akibatnya extract_title() memungut "NOMOR :" saja sebagai judul section
    (lihat metadata `section: "NOMOR :"` di corpus_regulasi_docling/), dan
    nomor keputusan tidak pernah menempel ke labelnya sehingga tidak bisa
    dicari lewat frasa "NOMOR 000.8.1.1/...". Pipeline PyMuPDF lama tidak
    kena ini karena membaca kedua kolom dalam satu baris.

    Label disatukan kembali dengan nilainya. Marker halaman
    PAGE_OPEN/PAGE_CLOSE sengaja TIDAK di-strip dari baris yang digabung --
    cuma dipakai untuk probe -- supaya pelacakan halaman tetap utuh.
    """
    lines = text.split("\n")
    joined = []
    i = 0

    while i < len(lines):
        probe = strip_page_marks(lines[i]).strip()

        if not ORPHAN_LABEL_RE.match(probe):
            joined.append(lines[i])
            i += 1
            continue

        j = i + 1
        while j < len(lines) and not strip_page_marks(lines[j]).strip():
            j += 1

        value = strip_page_marks(lines[j]).strip() if j < len(lines) else ""

        if (
            value
            and len(value) <= 80
            and not ORPHAN_LABEL_RE.match(value)
            and not BLOCK_HEADING_RE.match(value)
        ):
            joined.append(lines[i].strip() + " " + lines[j].strip())
            i = j + 1
            continue

        joined.append(lines[i])
        i += 1

    return "\n".join(_move_stray_lampiran_count(joined))


def _move_stray_lampiran_count(lines):
    """
    Di layout dua kolom itu, angka jumlah lampiran ("2") tercetak sebaris
    dengan nomor keputusan sehingga docling menempelkannya ke ekor NOMOR:

        NOMOR : 000.8.1.1/KEP.555-DISKOMINFO/2024 2
        LAMPIRAN : (DUA)

    Hasilnya nomor keputusan jadi salah kalau dikutip mentah. Angka di ekor
    NOMOR dipindahkan ke LAMPIRAN yang nilainya berupa ejaan dalam kurung
    ("(DUA)") -- kombinasi yang cuma muncul pada kasus ini, jadi baris NOMOR
    yang memang berakhir angka (mis. "NOMOR 290 TAHUN 2023") tidak ikut kena
    karena tidak ada LAMPIRAN "(...)" yang mengikutinya.
    """
    for idx, line in enumerate(lines):
        probe = strip_page_marks(line).strip()
        if not re.match(r"^NOMOR\s*:\s*\S.*\s+\d{1,2}$", probe, re.IGNORECASE):
            continue

        stray = re.search(r"(\d{1,2})$", probe).group(1)

        for k in range(idx + 1, min(idx + 7, len(lines))):
            probe_k = strip_page_marks(lines[k]).strip()
            if not re.match(r"^LAMPIRAN\s*:\s*\(", probe_k, re.IGNORECASE):
                continue

            lines[idx] = re.sub(r"\s+\d{1,2}\s*$", "", lines[idx])
            lines[k] = re.sub(
                r"(LAMPIRAN\s*:\s*)(\()",
                rf"\g<1>{stray} \g<2>",
                lines[k],
                count=1,
                flags=re.IGNORECASE,
            )
            break

    return lines


SURVIVING_NUM_RE = r"(?:\d{1,2}[.)]?\s+){0,2}"
SURVIVING_ALPHA_RE = r"(?:[a-z][.)]\s+)?"

PAGE_MARK_OPT = r"(?:%s\d+%s)?\s*" % (
    re.escape(PAGE_OPEN), re.escape(PAGE_CLOSE)
)

CONSIDERANS_SPLIT_RE = re.compile(
    r"[,;:]\s+(?=" + PAGE_MARK_OPT + SURVIVING_ALPHA_RE + r"bahwa\b)"
)

MENGINGAT_SPLIT_RE = re.compile(
    r"\)\s*[,;:]\s+(?=" + PAGE_MARK_OPT + SURVIVING_NUM_RE
    + r"(?:Undang[-\s]?Undang|Peraturan|Keputusan|Instruksi|Ketetapan)\b)"
)

BLOCK_TERMINATOR_RE = re.compile(
    r"^(?:MEMUTUSKAN|MENETAPKAN|Memutuskan|Menetapkan"
    r"|BAB\s+[IVXLC]+|LAMPIRAN\s+[IVXLC]+|Pasal\s+\d+)\b"
)

MIN_ITEM_LEN = 40
MAX_BLOCK_PARAGRAPHS = 40


def restore_enumeration_markers(text):
    """
    OCR docling menelan hampir semua penanda butir di blok preambul: di
    Perbup 290/2023 penanda "b." dan "c." pada Menimbang hilang total
    (butir b menyambung jadi satu kalimat dengan butir a), dan di Mengingat
    cuma "1." yang selamat -- butir 2 sampai belasan menyatu jadi satu
    paragraf raksasa. Padahal batangtubuhnya sendiri merujuk balik ke
    penanda itu ("sebagaimana dimaksud pada huruf a dan huruf b").

    Penanda direkonstruksi dari batas butir yang masih terbaca, lalu
    dipisah jadi paragraf sendiri supaya chunk_text() punya titik potong
    alami di antar-butir (bukan memotong di tengah kalimat). Titik koma
    yang hilang ikut dikembalikan di batas butir, karena di situlah
    posisinya pada naskah asli.

    Konservatif: relabel HANYA dijalankan kalau penanda pertama ("a." atau
    "1.") masih ada di teks OCR sebagai jangkar. Kalau jangkarnya ikut
    hilang, paragraf dibiarkan apa adanya -- lebih baik tidak bernomor
    daripada bernomor karangan.
    """
    paragraphs = text.split("\n\n")
    result = []
    i = 0

    while i < len(paragraphs):
        rebuilt = _try_rebuild_block(paragraphs, i)
        if rebuilt is None:
            result.append(paragraphs[i])
            i += 1
            continue
        block_text, next_index = rebuilt
        result.append(block_text)
        i = next_index

    return "\n\n".join(result)


LEADING_MARKS_RE = re.compile(r"^(?:\d+)*")


def _peel_marks(paragraph):
    """Pisahkan marker halaman di kepala paragraf dari isinya."""
    stripped = paragraph.strip()
    prefix = LEADING_MARKS_RE.match(stripped).group(0)
    return prefix, stripped[len(prefix):]


def _try_rebuild_block(paragraphs, start):
    _, head = _peel_marks(paragraphs[start])

    if re.match(r"^a\.\s+bahwa\b", head):
        return _rebuild_block(
            paragraphs,
            start,
            anchor_re=re.compile(r"^a\.\s+"),
            split_re=CONSIDERANS_SPLIT_RE,
            strip_re=re.compile(r"^(" + PAGE_MARK_OPT + r")" + SURVIVING_ALPHA_RE),
            label=lambda n: f"{chr(ord('a') + n)}.",
            heading="",
            extra_stop_re=re.compile(r"^Mengingat\b", re.IGNORECASE),
        )

    if re.match(r"^Mengingat\s*:?\s*1\.\s+\S", head, re.IGNORECASE):
        anchor_re = re.compile(r"^Mengingat\s*:?\s*", re.IGNORECASE)
        heading = "Mengingat :\n\n"
    elif re.match(r"^1\.\s+\S", head) and _follows_mengingat(paragraphs, start):
        anchor_re = re.compile(r"^")
        heading = ""
    else:
        return None

    return _rebuild_block(
        paragraphs,
        start,
        anchor_re=anchor_re,
        split_re=MENGINGAT_SPLIT_RE,
        strip_re=None,
        label=None,
        heading=heading,
        extra_stop_re=None,
    )


def _follows_mengingat(paragraphs, start):
    if start == 0:
        return False
    _, previous = _peel_marks(paragraphs[start - 1])
    return re.match(r"^Mengingat\s*:?\s*$", previous, re.IGNORECASE) is not None


def _rebuild_block(
    paragraphs, start, anchor_re, split_re, strip_re, label, heading,
    extra_stop_re,
):
    end = start + 1
    while end < len(paragraphs):
        if end - start > MAX_BLOCK_PARAGRAPHS:
            return None
        _, head = _peel_marks(paragraphs[end])
        if BLOCK_TERMINATOR_RE.match(head):
            break
        if extra_stop_re is not None and extra_stop_re.match(head):
            break
        end += 1

    prefix, first = _peel_marks(paragraphs[start])
    body = " ".join(
        [anchor_re.sub("", first, count=1)]
        + [p.strip() for p in paragraphs[start + 1:end]]
    )

    relabeled = _relabel(body, split_re, strip_re, label)
    if relabeled is None:
        return None

    return prefix + heading + relabeled, end


def _relabel(body, split_re, strip_re, label):
    """
    Pisahkan `body` jadi butir-butir. `label=None` berarti butir dipisah
    TANPA diberi penanda baru -- penanda bawaan teks dibiarkan apa adanya.

    Kembalikan None (bukan teks apa adanya) kalau pemisahan tidak
    meyakinkan, supaya pemanggil bisa mempertahankan paragraf ASLI lengkap
    dengan marker halamannya.
    """
    items = [item.strip() for item in split_re.split(body)]

    if strip_re is not None:
        items = [strip_re.sub(r"\g<1>", item) for item in items]

    items = [
        item + ")" if item.count("(") > item.count(")") else item
        for item in items
    ]

    if len(items) < 2 or any(len(item) < MIN_ITEM_LEN for item in items):
        return None

    if label is None:
        return ";\n\n".join(items)

    return ";\n\n".join(
        f"{label(n)} {item}" for n, item in enumerate(items)
    )


def clean_text(text):

    text = re.sub(r"\r", "\n", text)

    text = html.unescape(text)
    text = text.replace("\xa0", " ")

    text = fix_header_labels(text)

    text = re.sub(
        r"\b(BAB|LAMPIRAN)(\s+)(1+)\b",
        lambda m: f"{m.group(1)}{m.group(2)}{'I' * len(m.group(3))}",
        text,
    )

    text = re.sub(
        r"\bLAMPIRAN\s+(?=(?:PERATURAN|KEPUTUSAN)\b)",
        "LAMPIRAN I ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(r"^\s*-\s*\d+\s*-\s*$", "", text, flags=re.MULTILINE)
    text = strip_leaked_page_numbers(text)
    text = re.sub(r"^\s*[\ufffd•]\s*\d*\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"[ \t]+", " ", text)

    pasal_full_re = re.compile(r"^\s*Pasal\s+\d+", re.IGNORECASE)
    pasal_only_re = re.compile(r"^\s*Pasal\s*$", re.IGNORECASE)

    lines = text.split("\n")
    merged = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            if merged and merged[-1] != "":
                merged.append("")
            i += 1
            continue

        probe = strip_page_marks(stripped).strip()

        if pasal_full_re.match(probe):
            if merged and merged[-1] != "":
                merged.append("")
            merged.append(stripped)
            merged.append("")
            i += 1
            continue

        if pasal_only_re.match(probe):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and re.match(
                r"^\s*\d+\s*$", strip_page_marks(lines[j]).strip()
            ):
                combined = stripped + " " + lines[j].strip()
                if merged and merged[-1] != "":
                    merged.append("")
                merged.append(combined)
                merged.append("")
                i = j + 1
                continue

        if merged and merged[-1] and merged[-1] != "":
            merged[-1] = merged[-1] + " " + stripped
        else:
            merged.append(stripped)
        i += 1

    text = "\n\n".join(p for p in merged if p)

    marker_adjacent = re.compile(
        r"(\d+)(BAB\s+[IVXLC]+|LAMPIRAN\s+[IVXLC]+|Bagian\s+ke\s*\w+)",
        re.IGNORECASE,
    )
    text = marker_adjacent.sub(r"\n\n\1\2", text)

    text = re.sub(r"\s+(BAB\s+[IVXLC]+)", r"\n\n\1", text)
    text = re.sub(r"\s+(LAMPIRAN\s+[IVXLC]+)", r"\n\n\1", text)
    text = re.sub(
        r"\s+((?i:LAMPIRAN\s+[IVXLC]+\s*:?\s*Peraturan\s+Bupati))",
        r"\n\n\1",
        text,
    )
    text = re.sub(r"\s+(Bagian\s+ke\s*\w+)", r"\n\n\1", text, flags=re.IGNORECASE)

    text = re.sub(r"(Menimbang|Mengingat)\s*:\s*", r"\1 :\n\n", text)
    text = re.sub(r";\s*(?=\d{1,2}\.\s)", ";\n\n", text)
    text = re.sub(r";\s*(?=[a-z]\.\s)", ";\n\n", text)

    text = restore_enumeration_markers(text)

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"  +", " ", text)

    return text.strip()


def split_sections(text):

    pattern = (
        r"(?m)^[-\d]*\s*("
        r"BAB\s+[IVXLC]+[^\n]*|"
        r"Pasal\s+\d+[^\n]*|"
        r"LAMPIRAN\s+[IVXLC]+[^\n]*|"
        r"(?i:LAMPIRAN\s+[IVXLC]+\s*:?\s*Peraturan\s+Bupati)[^\n]*"
        r")"
    )
    matches = list(re.finditer(pattern, text, re.MULTILINE))

    sections = []

    if not matches:
        if len(strip_page_marks(text).strip()) > 100:
            return [text]
        return []

    pre_text = text[:matches[0].start()].strip()
    if len(strip_page_marks(pre_text).strip()) > 100:
        sections.append(pre_text)

    for i in range(len(matches)):
        start = matches[i].start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if len(strip_page_marks(section_text).strip()) > 50:
            sections.append(section_text)

    return sections


def fix_spaced_text(text):
    """Gabung huruf yang terpisah spasi dari PDF.
    Contoh: 'B U P A T I  B A N D U N G' -> 'BUPATI BANDUNG'
    Dipertahankan sebagai jaring pengaman -- kemungkinan kecil terjadi
    di output docling (beda dari PyMuPDF), tapi tidak merusak apapun
    kalau tidak match.
    """
    lines = text.split("\n")
    fixed_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            fixed_lines.append("")
            continue

        tokens = stripped.split()
        content_tokens = [t for t in tokens if t != "|"]
        single_chars = sum(1 for t in content_tokens if len(t) == 1)

        if (
            "|" not in tokens
            and len(tokens) > 2
            and single_chars / len(tokens) > 0.6
        ):
            words = re.split(r"\s{2,}", stripped)
            fixed_words = []
            for word in words:
                chars = word.split()
                if all(len(c) == 1 for c in chars) and len(chars) > 1:
                    fixed_words.append("".join(chars))
                else:
                    fixed_words.append(word)
            fixed_lines.append(" ".join(fixed_words))
        else:
            fixed_lines.append(stripped)

    return "\n".join(fixed_lines)


def sanitize_filename(name):
    name = str(name)
    name = re.sub(r'[\\/*?:"<>|]', "_", name)
    name = re.sub(r"\s+", "_", name)
    return name.strip()


def slugify(value):
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "regulasi"


def yaml_quote(value):
    value = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{value}"'


def infer_document_type(doc_name, text):
    upper_doc = doc_name.upper()
    upper_text = text.upper()
    if "KEPUTUSAN BUPATI" in upper_doc:
        return "keputusan_bupati"
    if "PERATURAN BUPATI" in upper_doc or "PERBUP" in upper_doc:
        return "perbup"
    if "PERATURAN BUPATI" in upper_text:
        return "perbup"
    if "PERATURAN DAERAH" in upper_text or "PERATURAN DAREAH" in upper_text:
        return "perda"
    return "regulasi_lain"


def chunk_text(text):
    """Pecah teks menjadi chunk dengan overlap yang dipaksa manual.

    Teks dipecah tanpa overlap, lalu CHUNK_OVERLAP karakter terakhir chunk
    sebelumnya ditempel di depan chunk berikutnya. Batas overlap digeser ke spasi
    terdekat agar yang ditempel selalu kata utuh.
    """
    if len(text) <= CHUNK_SIZE:
        return [text]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=0,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=False,
    )
    raw_chunks = [c.strip() for c in splitter.split_text(text) if c.strip()]

    min_standalone = CHUNK_OVERLAP * 2
    merged = []
    for chunk in raw_chunks:
        if merged and len(merged[-1]) < min_standalone:
            merged[-1] = f"{merged[-1]} {chunk}".strip()
        else:
            merged.append(chunk)
    if len(merged) > 1 and len(merged[-1]) < min_standalone:
        merged[-2] = f"{merged[-2]} {merged[-1]}".strip()
        merged.pop()

    chunks = []
    for i, chunk in enumerate(merged):
        if i == 0:
            chunks.append(chunk)
            continue
        overlap = merged[i - 1][-CHUNK_OVERLAP:]
        space_idx = overlap.find(" ")
        if space_idx != -1:
            overlap = overlap[space_idx + 1:]
        chunks.append(f"{overlap} {chunk}".strip())
    return chunks


def extract_title(text):
    text = strip_page_marks(text)
    lines = text.split("\n")
    for line in lines:
        candidate = line.strip()
        if (not candidate
                or re.match(r"^<!--.*-->$", candidate)
                or re.match(r"^-\s*\d+\s*-$", candidate)
                or re.match(r"^\d{1,3}$", candidate)
                or len(candidate) < 5):
            continue
        return candidate[:80]
    return ""


_LAST_PAGE = 1


def save_section(doc_name, source_file, idx, text, sub_idx=None, title=None):

    global _LAST_PAGE

    pages = sorted({int(n) for n in PAGE_RE.findall(text)})
    if pages:
        _LAST_PAGE = pages[-1]
    else:
        pages = [_LAST_PAGE]

    text = strip_page_marks(text)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()

    if not title:
        title = extract_title(text)
    if not title:
        title = f"SECTION_{idx}"

    suffix = f"_{sub_idx:02d}" if sub_idx is not None else ""
    filename = sanitize_filename(f"{doc_name}_{idx}_{title}{suffix}") + ".md"
    filepath = OUTPUT_DIR / filename

    document_type = infer_document_type(doc_name, text)
    source_id = slugify(doc_name)
    chunk_id = (
        f"{source_id}-{idx:03d}-{sub_idx:02d}"
        if sub_idx is not None
        else f"{source_id}-{idx:03d}"
    )

    halaman = "[" + ", ".join(str(p) for p in pages) + "]"

    markdown = f"""---
source_id: {yaml_quote(source_id)}
chunk_id: {yaml_quote(chunk_id)}
jenis: {yaml_quote("regulasi")}
tipe_dokumen: {yaml_quote(document_type)}
dokumen: {yaml_quote(doc_name)}
sumber_file: {yaml_quote(source_file)}
halaman: {halaman}
section: {yaml_quote(title)}
---

# {title}

{text}
"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(markdown)


def clear_regulasi_files():
    for path in OUTPUT_DIR.glob("*.md"):
        if path.name.startswith("SOP_Pengaduan_"):
            continue
        path.unlink()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bangun korpus regulasi+SOP dari docling, dengan opsi "
                     "fallback vision (Claude/GPT-4o) untuk halaman bertabel/"
                     "bergambar yang terdeteksi otomatis oleh docling."
    )
    parser.add_argument(
        "--model", choices=["claude", "gpt4o", "none"], default=VISION_PROVIDER,
        help="Provider vision untuk halaman tabel/gambar yang terdeteksi "
             "docling. 'none' = matikan fallback vision sama sekali, murni "
             f"docling+Tesseract seperti sebelum ada fitur ini. (default: {VISION_PROVIDER})"
    )
    return parser.parse_args()


def main():

    global _LAST_PAGE, VISION_PROVIDER

    args = parse_args()
    VISION_PROVIDER = args.model
    if VISION_PROVIDER == "none":
        print("[INFO] Vision fallback DIMATIKAN (--model none) -- murni docling+Tesseract.")
    else:
        print(f"[INFO] Vision fallback aktif: provider={VISION_PROVIDER}")

    clear_regulasi_files()
    print("[CLEAR] File di corpus_regulasi_sop/ dihapus")

    pdf_files = list(PDF_DIR.glob("*.pdf"))
    print(f"[INFO] Total PDF: {len(pdf_files)}")

    converter = build_converter()

    for pdf_path in pdf_files:
        if "SOP" in pdf_path.name.upper():
            print(f"\n[SKIP] {pdf_path.name}")
            continue

        print(f"\n[PROCESS] {pdf_path.name}")

        _LAST_PAGE = 1

        raw_text = extract_pdf(converter, pdf_path)
        cleaned = clean_text(raw_text)
        sections = split_sections(cleaned)

        print(f"[INFO] Total section: {len(sections)}")

        doc_name = pdf_path.stem

        for idx, section in enumerate(sections):
            chunks = chunk_text(section)

            if len(chunks) == 1:
                save_section(doc_name, pdf_path.name, idx, chunks[0])
            else:
                section_title = extract_title(chunks[0])
                for sub_idx, chunk in enumerate(chunks):
                    save_section(
                        doc_name, pdf_path.name, idx, chunk, sub_idx,
                        title=section_title
                    )

        print("[DONE]")

    print("\n[SUCCESS] Semua corpus regulasi selesai")

if __name__ == "__main__":
    main()