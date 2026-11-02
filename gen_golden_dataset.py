import argparse
import glob
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import urlopen, Request

def _load_dotenv():
    for p in [Path(__file__).resolve().parent / ".env", Path.cwd() / ".env"]:
        if p.exists():
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    k = k.strip(); v = v.strip().strip('"').strip("'")
                    if k and k not in os.environ:
                        os.environ[k] = v
            print(f"[INFO] .env dimuat dari {p}")
            return
_load_dotenv()


MIN_CHUNK_LENGTH = 100
BOILERPLATE_PATTERNS = [
    r"^(daftar isi|daftar pustaka|kata pengantar)",
    r"^\s*\.\s*\.\s*\.",
    r"^(halaman|page)\s+\d+",
]
DEDUP_THRESHOLD = 0.85

TARGET_DIST = {
    "regulasi": {
        "answer_fact": 0.45,
        "answer_list": 0.40,
        "answer_procedure": 0.10,
        "multi_hop": 0.05,
    },
    "pengaduan": {
        "answer_fact": 0.30,
        "answer_procedure": 0.30,
        "answer_temporal": 0.15,
        "multi_hop": 0.15,
        "answer_list": 0.10,
    },
}


def parse_md_file(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read().replace("\r\n", "\n")
    parts = raw.split("---")
    if len(parts) < 3:
        return None
    meta = {}
    for line in parts[1].strip().split("\n"):
        m = re.match(r'^([a-zA-Z_]+):\s*"?(.*?)"?\s*$', line.strip())
        if m:
            meta[m.group(1)] = m.group(2).strip()
    body = "---".join(parts[2:]).strip()
    chunk_id = meta.get("chunk_id")
    if not chunk_id or not body:
        return None
    return {"meta": meta, "body": body, "chunk_id": chunk_id}


def is_boilerplate(body):
    body_lower = body.lower().strip()
    for pattern in BOILERPLATE_PATTERNS:
        if re.match(pattern, body_lower):
            return True
    return False


TINDAK_LANJUT_BOILERPLATE = [
    "tidak menerangkan nik",
    "tidak dapat kami proses",
    "tidak dapat diproses",
    "tidak dapat melakukan pengecekan",
    "silahkan untuk menerangkan",
    "mohon melengkapi",
    "tidak melampirkan",
    "tidak lengkap",
    "tidak dapat ditindaklanjuti",
    "tidak memenuhi",
    "laporan tersebut tidak menerangkan",
]


def extract_tindak_lanjut_from_body(body):
    """Ambil potongan teks setelah penanda [TINDAK LANJUT ...] pada body chunk."""
    m = re.search(r"\[TINDAK LANJUT", body, re.IGNORECASE)
    if not m:
        return ""
    return body[m.start():].strip()


def extract_masalah_from_body(body):
    """Ambil teks bagian [MASALAH] sampai sebelum [TINDAK LANJUT]."""
    m = re.search(r"\[MASALAH\]", body, re.IGNORECASE)
    if not m:
        return ""
    start = m.end()
    end = re.search(r"\[TINDAK LANJUT", body[start:], re.IGNORECASE)
    return (body[start:start + end.start()] if end else body[start:]).strip()


RESPON_AWAL_MARKERS = [
    "akan kami sampaikan ke unit",
    "akan segera kami tindaklanjuti sesuai batas waktu",
    "sesuai batas waktu tindak lanjut di sp4n",
    "sesuai batas waktu tindaklanjut di sp4n",
]


def has_substantive_tindak_lanjut(chunk, min_len=120):
    """True jika tindak lanjut aduan cukup substantif untuk dijadikan sumber
    soal: panjangnya memadai DAN bukan sekadar template penutupan/minta data
    atau template respon awal (acknowledgment).
    Selaras dengan analisis: ~2965 dari 3358 aduan lolos kriteria ini."""
    tl = extract_tindak_lanjut_from_body(chunk.get("body", ""))
    if len(tl) < min_len:
        return False
    low = tl.lower()
    if any(p in low for p in TINDAK_LANJUT_BOILERPLATE):
        return False
    tl_count = len(re.findall(r"\[TINDAK LANJUT", tl, re.IGNORECASE))
    if tl_count <= 1 and any(p in low for p in RESPON_AWAL_MARKERS):
        return False
    return True


def load_corpus(regulasi_dir, md_dir):
    reg = []
    for path in glob.glob(os.path.join(regulasi_dir, "*.md")):
        item = parse_md_file(path)
        if item and len(item["body"]) >= MIN_CHUNK_LENGTH and not is_boilerplate(item["body"]):
            reg.append(item)

    peng_raw = []
    for path in glob.glob(os.path.join(md_dir, "*.md")):
        item = parse_md_file(path)
        if item:
            peng_raw.append(item)

    grouped = defaultdict(dict)
    for item in peng_raw:
        sid = item["meta"].get("source_id")
        idx = item["meta"].get("chunk_index", "0")
        if sid:
            grouped[sid][idx] = item

    peng = {
        sid: chunks for sid, chunks in grouped.items()
        if "0" in chunks and len(chunks["0"]["meta"].get("judul", "")) >= 5
        and chunks["0"]["meta"].get("instansi")
        and chunks["0"]["meta"].get("status")
    }

    return reg, peng


def load_all_chunk_ids(regulasi_dir, md_dir):
    """Enumerasi semua chunk_id yang ada di korpus produksi, tanpa filter
    kelayakan yang dipakai load_corpus(). Dipakai untuk mengecek keberadaan
    chunk_id, bukan untuk memilih kandidat soal baru.

    Untuk folder pengaduan, chunk_id diambil dari nama file bila konvensi
    chunk_id == nama file terpenuhi pada sampel. Bila tidak, fungsi beralih ke
    parsing frontmatter setiap file.
    """
    ids = set()

    for path in glob.glob(os.path.join(regulasi_dir, "*.md")):
        item = parse_md_file(path)
        if item:
            ids.add(item["chunk_id"])

    md_paths = glob.glob(os.path.join(md_dir, "*.md"))
    sample = md_paths[:20]
    fast_path_ok = bool(sample) and all(
        re.match(r"^pengaduan-.+-\d{2}$", Path(p).stem) for p in sample
    )
    if fast_path_ok:
        ids.update(Path(p).stem for p in md_paths)
    else:
        for path in md_paths:
            item = parse_md_file(path)
            if item:
                ids.add(item["chunk_id"])

    return ids


def validate_existing_dataset(dataset_path, regulasi_dir, md_dir):
    """Mode backfill/maintenance -- TIDAK memanggil Claude API sama sekali dan
    TIDAK mengubah isi pertanyaan/jawaban sedikit pun.

    Ini cara generator "tersambung" dengan golden dataset final yang sudah ada
    (mis. golden_dataset_fix.jsonl, sudah divalidasi manual + diverifikasi
    Diskominfo): memvalidasi ulang setiap item terhadap korpus produksi
    terkini, supaya kalau korpus pernah di-rebuild dan chunk_id berubah,
    driftnya kelihatan lebih awal -- bukan menciptakan ulang datasetnya.
    """
    valid_ids = load_all_chunk_ids(regulasi_dir, md_dir)
    items = [json.loads(l) for l in open(dataset_path, encoding="utf-8")]

    required_fields = ("question", "ground_truth", "behavior")
    report = {"ok": [], "missing_chunk": [], "missing_field": [], "no_expected_chunk": []}

    for it in items:
        item_id = it.get("id")
        missing_fields = [f for f in required_fields if not it.get(f)]
        if missing_fields:
            report["missing_field"].append((item_id, missing_fields))
            continue

        expected = it.get("expected_chunk_ids", [])
        if not expected:
            report["no_expected_chunk"].append(item_id)
            continue

        bad = [c for c in expected if c not in valid_ids]
        if bad:
            report["missing_chunk"].append((item_id, bad))
        else:
            report["ok"].append(item_id)

    return items, report


def print_validation_report(report, total):
    print(f"\n{'='*60}")
    print(f"VALIDASI DATASET FINAL vs KORPUS PRODUKSI (tanpa panggil API)")
    print(f"{'='*60}")
    print(f"Total item             : {total}")
    print(f"OK (chunk_id valid)    : {len(report['ok'])}")
    print(f"Tanpa expected_chunk   : {len(report['no_expected_chunk'])} (wajar utk not_found/agregasi)")
    print(f"Field wajib kosong     : {len(report['missing_field'])}")
    print(f"Chunk_id HILANG        : {len(report['missing_chunk'])}")

    if report["missing_chunk"]:
        print("\n  Item dengan chunk_id yang sudah tidak ada di korpus (perlu remap manual):")
        for item_id, bad in report["missing_chunk"]:
            print(f"    id={item_id}: {bad}")

    if report["missing_field"]:
        print("\n  Item dengan field wajib kosong:")
        for item_id, fields in report["missing_field"]:
            print(f"    id={item_id}: {fields}")


SYSTEM_REGULASI = """Kamu adalah peneliti yang membuat dataset evaluasi untuk sistem RAG berbasis regulasi pemerintah Kabupaten Bandung.

Dari teks chunk regulasi yang diberikan, generate 1 pasangan pertanyaan-jawaban evaluasi.

KRITERIA WAJIB:
1. Pertanyaan HARUS natural — seperti cara orang benar-benar bertanya, bukan gaya query database
2. Pertanyaan WAJIB menyebut nama peraturan secara SPESIFIK (mis. "Peraturan Bupati Nomor 38 Tahun 2005" atau "Perbup No. 38/2005", beserta Pasal/Lampiran bila ada). DILARANG memakai rujukan kontekstual seperti "peraturan ini", "dokumen ini", "aturan tersebut" — karena pertanyaan harus bisa dijawab MANDIRI tanpa konteks tambahan (pengguna tidak tahu "ini" merujuk ke apa).
3. Jawaban HARUS extractive — diambil langsung dari teks, bukan interpretasi/pengetahuan umum
4. Jawaban harus lengkap dan presisi (mis. sebutkan tanggal/angka yang eksplisit, bukan "pada tanggal ditetapkan")
5. Pilih tipe SESUAI instruksi yang diberikan di prompt (bukan pilihan bebas)
6. HINDARI pertanyaan bernilai rendah: nama pejabat penanda tangan, tanggal berlaku yang hanya berbunyi "pada tanggal ditetapkan/diundangkan", atau nomor lembaran. Jika chunk HANYA berisi hal seperti itu (mis. klausul penutup/tanda tangan), lebih baik hasilkan pertanyaan tentang isi substantif yang tetap ada di chunk. PRIORITASKAN: definisi, syarat/ketentuan, prosedur/tahapan, batas waktu, kewenangan/tugas, sanksi.

Penjelasan tipe:
- answer_fact: tanya 1 fakta/definisi/angka spesifik (contoh: "Berapa hari batas waktu X?", "Apa definisi Y?")
- answer_list: ada enumerasi di teks dengan kata "antara lain/meliputi/terdiri dari" → tanya semua itemnya (contoh: "Apa saja unsur X?")
- answer_procedure: ada langkah/tahapan berurutan → tanya alurnya (contoh: "Bagaimana alur/prosedur X?")
- multi_hop: tanya 2 hal sekaligus yang keduanya ada di teks (contoh: "Siapa yang berwenang X dan berapa lama batas waktunya?")

Output HANYA JSON valid ini (tanpa backtick, tanpa komentar):
{"question": "...", "ground_truth": "...", "expected_keyword": ["kw1", "kw2"], "behavior": "answer_fact"}"""

SYSTEM_PENGADUAN = """Kamu adalah peneliti yang membuat dataset evaluasi untuk sistem RAG berbasis data pengaduan masyarakat Kabupaten Bandung.

Dari isi aduan (masalah + TINDAK LANJUT nyata) yang diberikan, generate 1 pertanyaan natural + jawaban acuan sesuai tipe yang diminta.

PRINSIP UTAMA:
- PERTANYAAN dibuat dari MASALAH yang dilaporkan warga (natural, seperti warga bertanya).
- JAWABAN (ground_truth) diambil dari ISI TINDAK LANJUT yang benar-benar dilakukan instansi -- APA yang dikerjakan/diarahkan, BUKAN sekadar status "Ditutup oleh Sistem".
- Jawaban HARUS setia pada teks tindak lanjut (extractive/parafrase dekat), DILARANG mengarang tindakan yang tidak tertulis.

KRITERIA WAJIB:
1. JANGAN tanya tentang "klasifikasi" — field internal yang tidak terlihat pengguna.
2. Gunakan JUDUL/isi masalah di pertanyaan, bukan nomor ID.
3. Pertanyaan natural seperti di chat/WhatsApp.
4. Untuk sebagian besar tipe, jawaban bersumber dari tindak lanjut; metadata (status, instansi, tanggal, durasi) sebagai pelengkap. KHUSUS tipe answer_temporal, metadata waktu/status JUSTRU menjadi inti jawaban.

Penjelasan tipe:
- answer_fact: tanya 1 fakta konkret dari tindak lanjut/penanganan (mis. "aduan X ditindaklanjuti bagaimana?", "aduan X diteruskan ke mana?").
- answer_procedure: tanya ALUR/langkah tindak lanjut yang dilakukan instansi atas aduan tersebut.
- answer_list: bila tindak lanjut memuat beberapa langkah/arahan, tanya seluruh langkahnya.
- multi_hop: gabungkan masalah + tindak lanjut (mis. "apa masalahnya dan bagaimana ditangani?").
- answer_temporal: tanya WAKTU penanganan aduan -- kapan masuk, berapa lama ditindaklanjuti/diselesaikan, atau statusnya. Untuk tipe INI ground_truth diambil dari metadata tanggal/durasi/status yang diberikan, dengan angka/tanggal EKSPLISIT (mis. "Aduan tercatat masuk pada 2 Januari 2020, mulai ditindaklanjuti setelah 8 hari, dan berstatus Ditutup oleh Admin."). JANGAN mengarang tanggal/durasi yang tidak ada di metadata; jika metadata waktu kosong, pilih tipe lain.

CATATAN: kalau di bawah ada instruksi "CATATAN PENTING" yang merujuk ke
pertanyaan/jawaban yang sudah ada, instruksi itu WAJIB diikuti persis --
fakta yang dipakai harus SAMA seperti yang dirujuk di sana.

Output HANYA JSON valid ini:
{"question": "...", "ground_truth": "...", "expected_keyword": ["kw1", "kw2"], "behavior": "answer_fact"}"""

SYSTEM_NOT_FOUND = """Kamu adalah peneliti yang membuat dataset evaluasi untuk sistem RAG pengaduan
Kabupaten Bandung, KHUSUS kategori "di luar cakupan" (not_found) -- soal yang jawabannya
SENGAJA tidak tersedia di basis pengetahuan sistem, untuk menguji apakah sistem menolak
dengan jujur (bukan mengarang jawaban).

KRITERIA WAJIB:
1. Pertanyaan harus terdengar natural, seperti pertanyaan warga sungguhan -- bukan
   pertanyaan yang jelas aneh/mengada-ada.
2. ground_truth WAJIB berpola: "informasi ini tidak tersedia dalam basis pengetahuan
   sistem ini" DITAMBAH arahan yang membantu ke pihak/kanal yang benar-benar relevan --
   BUKAN sekadar menolak mentah. Contoh pola yang benar: pertanyaan soal SIM -> arahkan
   ke Satpas/Polres/Samsat (bukan Diskominfo Kab. Bandung); pertanyaan soal wilayah lain
   -> arahkan cek ke pemda wilayah tsb; pertanyaan nomor tracking fiktif -> arahkan cek
   ulang di aplikasi LAPOR! atau hubungi instansi terkait.
3. Variasikan jenis "tidak ditemukan"-nya -- pilih SALAH SATU pola per soal:
   a. Wilayah di luar cakupan (bukan Kabupaten Bandung)
   b. Layanan/urusan yang memang bukan wewenang Pemkab Bandung/Diskominfo (mis. SIM,
      paspor, pajak kendaraan, KTP -- itu instansi lain, arahkan ke sana)
   c. Referensi pasal/dokumen/nomor SK yang TIDAK ADA -- HARUS benar-benar tidak eksis.
      Kalau ragu berapa pasal terakhir suatu dokumen, pilih nomor yang JAUH melebihi
      angka wajar (mis. lebih dari 2x pasal terakhir yang biasa ada di perda/perbup)
      supaya besar kemungkinan memang tidak ada, bukan cuma sedikit di luar jangkauan.
   d. Nomor tracking/ID pengaduan fiktif yang tidak mungkin ada (pakai pola angka acak
      generik, JANGAN meniru pola nomor pengaduan asli Kabupaten Bandung)
4. JANGAN sertakan detail teknis spesifik di ground_truth yang sebenarnya tidak kamu
   ketahui pasti kebenarannya -- cukup "tidak tersedia + arahan", jangan menjelaskan
   ISI dari sesuatu yang kamu klaim tidak ada.

Output HANYA JSON valid ini (tanpa backtick, tanpa komentar):
{"question": "...", "ground_truth": "...", "expected_keyword": ["kw1", "kw2"],
 "source_type": "regulasi atau pengaduan", "claim_ref": "referensi spesifik yang
 diklaim tidak ada kalau ada, mis. 'Pasal 60 Perbup No 290 Tahun 2023' atau nomor
 tracking yang disebut -- null kalau soal berbasis wilayah/layanan di luar cakupan
 (bukan referensi dokumen/nomor spesifik)"}"""

SYSTEM_VARIASI = """Buat 3 variasi gaya bahasa dari pertanyaan berikut. Makna/topik TIDAK boleh berubah.

Gaya yang dibutuhkan:
1. "santai" — bahasa percakapan informal tapi jelas
2. "typo_singkatan" — ada typo kecil dan singkatan umum (spt, gmn, brp, tdk, dll, yg)
3. "sunda_campuran" — campur Sunda dan Indonesia (teh, mah, atuh, kumaha, naon, kedah, sareng)

Output HANYA JSON array:
[{"gaya": "santai", "question": "..."}, {"gaya": "typo_singkatan", "question": "..."}, {"gaya": "sunda_campuran", "question": "..."}]"""


def call_claude(prompt, system="", temperature=0.3, max_tokens=900, model="claude-sonnet-4-6", _retry=True):
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY belum di-set.")

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        payload["system"] = system

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
    with urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())
    raw = result["content"][0]["text"].strip()
    stop_reason = result.get("stop_reason")

    if stop_reason == "max_tokens" and _retry:
        return call_claude(prompt, system, temperature, max_tokens=min(max_tokens * 2, 4096),
                            model=model, _retry=False)

    if raw and "{" not in raw:
        if _retry:
            return call_claude(prompt, system, temperature, max_tokens, model=model, _retry=False)
        snippet = raw[:200].replace("\n", " ")
        raise RuntimeError(
            f"Respons Claude bukan JSON (tidak ada '{{' sama sekali) 2x berturut-turut, "
            f"kemungkinan model menolak/menjelaskan alih-alih menjawab. "
            f"Cuplikan respons: {snippet!r}"
        )

    raw = re.sub(r"```json?\s*", "", raw)
    raw = re.sub(r"```\s*", "", raw).strip()
    return raw


def call_gpt4o(prompt, system="", temperature=0.3, max_tokens=900, model="gpt-4o", _retry=True):
    """Setara call_claude() tapi ke OpenAI Chat Completions -- dipakai kalau
    --model diisi nama model GPT (lihat call_llm()). Kontrak return SAMA
    PERSIS dengan call_claude(): teks mentah tanpa backtick, siap masuk ke
    safe_parse_json()."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY belum di-set.")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
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
    with urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())
    choice = result["choices"][0]
    raw = (choice["message"].get("content") or "").strip()
    finish_reason = choice.get("finish_reason")

    if finish_reason == "length" and _retry:
        return call_gpt4o(prompt, system, temperature, max_tokens=min(max_tokens * 2, 4096),
                           model=model, _retry=False)

    if not raw:
        if _retry:
            return call_gpt4o(prompt, system, temperature, max_tokens, model=model, _retry=False)
        raise RuntimeError(
            f"OpenAI mengembalikan content kosong 2x berturut-turut "
            f"(finish_reason={finish_reason!r})."
        )

    if "{" not in raw:
        if _retry:
            return call_gpt4o(prompt, system, temperature, max_tokens, model=model, _retry=False)
        snippet = raw[:200].replace("\n", " ")
        raise RuntimeError(
            f"Respons OpenAI bukan JSON (tidak ada '{{' sama sekali) 2x berturut-turut, "
            f"kemungkinan model menolak/menjelaskan alih-alih menjawab. "
            f"Cuplikan respons: {snippet!r}"
        )

    raw = re.sub(r"```json?\s*", "", raw)
    raw = re.sub(r"```\s*", "", raw).strip()
    return raw


def call_llm(prompt, system="", temperature=0.3, max_tokens=900, model="claude-sonnet-4-6", _retry=True):
    """Pilih provider berdasarkan nama model: diawali 'gpt' memakai OpenAI,
    selain itu memakai Claude (default).
    """
    if model.lower().startswith("gpt"):
        return call_gpt4o(prompt, system, temperature, max_tokens, model=model, _retry=_retry)
    return call_claude(prompt, system, temperature, max_tokens, model=model, _retry=_retry)


def required_api_key_name(model):
    """Nama env var API key yang dibutuhkan untuk --model tertentu -- dipakai
    di gate pengecekan CLI supaya milih model gpt-* tidak lagi keliru minta
    ANTHROPIC_API_KEY (dan sebaliknya)."""
    return "OPENAI_API_KEY" if model.lower().startswith("gpt") else "ANTHROPIC_API_KEY"


def safe_parse_json(raw):
    """Parse JSON dengan toleransi terhadap output LLM yang tidak sempurna
    (mis. ada teks pembuka/penutup di luar objek JSON). Truncation akibat
    max_tokens sudah ditangani di call_claude(), jadi di sini raw seharusnya
    sudah lengkap -- fallback ini hanya untuk kasus format, bukan potongan."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            return json.loads(m.group())
        snippet = raw[:200].replace("\n", " ")
        raise ValueError(f"Respons bukan JSON valid: {e}. Cuplikan: {snippet!r}") from None


def _build_reference_anchor(reference_qa):
    """Bila reference_qa diisi (mode --consistency-check), tambahkan instruksi
    agar model menyusun ulang fakta yang sama persis, bukan memilih fakta lain
    dari chunk yang sama.
    """
    if not reference_qa:
        return ""
    ref_q, ref_a = reference_qa
    return f"""

CATATAN PENTING: Pasangan QA berikut SUDAH ADA dan sudah diverifikasi manual,
dari chunk yang SAMA:
  Pertanyaan lama: "{ref_q}"
  Jawaban lama: "{ref_a}"
Tugas Anda BUKAN membuat pertanyaan tentang fakta lain dari chunk ini --
buat ULANG pasangan QA yang MERUJUK FAKTA YANG SAMA PERSIS seperti di atas.
Boleh beda kata/kalimat/gaya bahasa, tapi maknanya harus setara -- jangan
pilih fakta lain yang kebetulan juga ada di chunk yang sama."""


def readable_doc_name(meta):
    """Ubah slug dokumen (mis. 'No-38-Tahun-2005') jadi nama peraturan yang
    layak disebut di pertanyaan, memakai tipe_dokumen + nomor/tahun. Kalau
    field dokumen sudah berupa nama deskriptif (mengandung 'tentang'/banyak
    kata), pakai apa adanya. Tujuannya: pertanyaan bisa menyebut peraturan
    secara spesifik & mandiri, bukan 'peraturan ini'."""
    dok = str(meta.get("dokumen", "")).strip()
    tipe = str(meta.get("tipe_dokumen", "")).strip().lower()
    if dok and (" tentang " in dok.lower() or len(dok.split()) >= 4):
        return dok
    m = re.search(r"no[\-\s]*(\d+)[\-\s]*(?:tahun|thn)[\-\s]*(\d{4})", dok, re.IGNORECASE)
    nomor_tahun = f"Nomor {m.group(1)} Tahun {m.group(2)}" if m else dok.replace("-", " ").strip()
    prefix = {
        "perbup": "Peraturan Bupati Kabupaten Bandung",
        "perda": "Peraturan Daerah Kabupaten Bandung",
        "keputusan_bupati": "Keputusan Bupati Kabupaten Bandung",
        "sop": "Standar Operasional Prosedur",
    }.get(tipe, "Peraturan")
    return f"{prefix} {nomor_tahun}".strip()


def gen_regulasi(chunk, log, model="claude-sonnet-4-6", target_behavior="answer_fact", reference_qa=None):
    body = chunk["body"][:2500]
    meta = chunk["meta"]
    anchor = _build_reference_anchor(reference_qa)
    nama_dokumen = readable_doc_name(meta)
    prompt = f"""Dokumen: {nama_dokumen}
Section: {meta.get('section', '')}
(WAJIB sebut nama dokumen di atas secara spesifik pada pertanyaan; DILARANG memakai "peraturan ini"/"dokumen ini".)

Teks chunk:
{body}

INSTRUKSI: Generate 1 pasangan QA dengan tipe behavior = "{target_behavior}".
Pastikan tipe di output JSON sesuai instruksi ini.{anchor}"""

    raw = call_llm(prompt, SYSTEM_REGULASI, temperature=0.3, model=model)
    qa = safe_parse_json(raw)
    qa["behavior"] = target_behavior
    log["raw_output"] = raw
    log["behavior"] = target_behavior
    return qa


def gen_pengaduan(source_id, chunks, log, model="claude-sonnet-4-6", target_behavior="answer_fact", reference_qa=None, extra_context=None):
    c0 = chunks.get("0") or chunks.get(0)
    if not c0:
        return None
    meta = c0["meta"]
    anchor = _build_reference_anchor(reference_qa)
    extra = f"\n\nKonteks tambahan (kutipan tindak lanjut terkait):\n{extra_context}" if extra_context else ""
    body = c0.get("body", "")
    masalah = extract_masalah_from_body(body)
    tindak_lanjut = extract_tindak_lanjut_from_body(body)
    prompt = f"""Isi aduan:
- Judul: {meta.get('judul', '')}
- Masalah yang dilaporkan: {masalah}
- TINDAK LANJUT (yang benar-benar dilakukan instansi): {tindak_lanjut}

Metadata pelengkap (boleh dipakai jika relevan, bukan inti jawaban):
- Instansi: {meta.get('instansi', '')}
- Status: {meta.get('status', '')}
- Kategori: {meta.get('kategori', '')}
- Kecamatan: {meta.get('kecamatan', '')}
- Tanggal masuk: {meta.get('tanggal_masuk', meta.get('tahun', ''))}
- Durasi respon (hari): {meta.get('durasi_respon', '')}
- Durasi penyelesaian (hari): {meta.get('durasi_penyelesaian', '')}
{extra}

INSTRUKSI: Generate 1 pertanyaan natural + jawaban acuan dengan tipe behavior = "{target_behavior}".
JAWABAN (ground_truth) HARUS bersumber dari TINDAK LANJUT di atas (apa yang dilakukan instansi), bukan sekadar status. Jangan mengarang.
Pastikan tipe di output JSON sesuai instruksi ini.{anchor}"""

    raw = call_llm(prompt, SYSTEM_PENGADUAN, temperature=0.3, model=model)
    qa = safe_parse_json(raw)
    qa["behavior"] = target_behavior
    log["raw_output"] = raw
    log["behavior"] = target_behavior
    return qa


def _gen_regulasi_with_fallback(chunk, target_beh, model):
    """Coba generate dgn target_beh; kalau model MENOLAK (bukan error teknis --
    lihat call_gpt4o()/call_claude(): chunk tidak cocok dgn tipe soal yg diminta,
    mis. diminta 'answer_list' tapi isinya bukan enumerasi), JANGAN langsung buang
    chunk-nya. Retry SEKALI lagi dgn 'answer_fact' (tipe paling universal, cocok
    hampir semua chunk) sebelum benar-benar menyerah -- chunk yang tadinya sia-sia
    jadi tetap kepakai. Kalau target_beh awalnya SUDAH answer_fact, tidak ada
    gunanya retry (pasti gagal dgn alasan sama), langsung menyerah.
    Return: (qa_or_None, behavior_yang_akhirnya_dipakai_or_None, [log, ...])."""
    log = {"step": "regulasi", "chunk_id": chunk["chunk_id"], "target_behavior": target_beh, "status": ""}
    try:
        qa = gen_regulasi(chunk, log, model=model, target_behavior=target_beh)
        return qa, target_beh, [log]
    except Exception as e:
        log["status"] = "ERROR"; log["error"] = str(e)
        print(f"  [ERROR] {chunk['chunk_id'][:35]}: {e}")
        if target_beh == "answer_fact":
            return None, None, [log]
        log2 = {"step": "regulasi", "chunk_id": chunk["chunk_id"], "target_behavior": "answer_fact",
                 "status": "", "retry_after_reject": target_beh}
        try:
            qa = gen_regulasi(chunk, log2, model=model, target_behavior="answer_fact")
            return qa, "answer_fact", [log, log2]
        except Exception as e2:
            log2["status"] = "ERROR"; log2["error"] = str(e2)
            print(f"  [ERROR] {chunk['chunk_id'][:35]} (retry answer_fact): {e2}")
            return None, None, [log, log2]


def _gen_pengaduan_with_fallback(sid, chunks, target_beh, model):
    """Sama seperti _gen_regulasi_with_fallback(), versi pengaduan."""
    log = {"step": "pengaduan", "source_id": sid, "target_behavior": target_beh, "status": ""}
    try:
        qa = gen_pengaduan(sid, chunks, log, model=model, target_behavior=target_beh)
        return qa, target_beh, [log]
    except Exception as e:
        log["status"] = "ERROR"; log["error"] = str(e)
        print(f"  [ERROR] sid={sid}: {e}")
        if target_beh == "answer_fact":
            return None, None, [log]
        log2 = {"step": "pengaduan", "source_id": sid, "target_behavior": "answer_fact",
                 "status": "", "retry_after_reject": target_beh}
        try:
            qa = gen_pengaduan(sid, chunks, log2, model=model, target_behavior="answer_fact")
            return qa, "answer_fact", [log, log2]
        except Exception as e2:
            log2["status"] = "ERROR"; log2["error"] = str(e2)
            print(f"  [ERROR] sid={sid} (retry answer_fact): {e2}")
            return None, None, [log, log2]


def find_regulasi_chunk(chunk_id, regulasi_dir):
    """Cari 1 chunk regulasi spesifik by chunk_id. Folder regulasi_sop kecil
    (~239 file) jadi scan penuh tetap cepat."""
    for path in glob.glob(os.path.join(regulasi_dir, "*.md")):
        item = parse_md_file(path)
        if item and item["chunk_id"] == chunk_id:
            return item
    return None


def find_pengaduan_chunk0(source_id, md_dir):
    """Baca langsung file chunk index 0 untuk source_id tertentu, memakai
    konvensi build_corpus_aduan.py (chunk_id = f"pengaduan-{source_id}-00",
    filename = chunk_id + ".md") -- lebih cepat daripada scan seluruh folder."""
    path = Path(md_dir) / f"pengaduan-{source_id}-00.md"
    if not path.exists():
        return None
    return parse_md_file(str(path))


def run_consistency_check(dataset_path, regulasi_dir, md_dir, model, delay, limit, out_path, only_ids=None):
    """Mode PEMBANDING SEMANTIK -- bukan pengganti dataset final.

    Untuk tiap item di dataset final yang punya sumber chunk jelas (regulasi
    atau pengaduan, behavior != not_found), generate ULANG kandidat lewat
    Claude API dari chunk sumber yang SAMA + kriteria behavior yang sama,
    lalu simpan berdampingan dengan versi final -- supaya bisa dinilai manual
    apakah maknanya konsisten walau redaksi berbeda.

    PENTING:
    - dataset_path (mis. golden_dataset_fix.jsonl) TIDAK PERNAH ditulis/diubah
      oleh fungsi ini. Hasil perbandingan ditulis ke file laporan terpisah.
    - Kalau setelah lihat laporan ini Anda memutuskan mengedit redaksi salah
      satu item di dataset final, item yang REDAKSINYA benar-benar berubah
      itu yang idealnya dicek ulang ke Diskominfo -- bukan berarti seluruh
      100 item perlu diverifikasi ulang, karena yang lain tidak disentuh.
    """
    items = [json.loads(l) for l in open(dataset_path, encoding="utf-8")]
    items = [it for it in items if it.get("behavior") != "not_found"]
    if only_ids:
        items = [it for it in items if it.get("id") in only_ids]
    if limit:
        items = items[:limit]

    print(f"\nMenjalankan pembanding semantik untuk {len(items)} item "
          f"(model={model}) -- setiap item = 1 panggilan Claude API.")

    results = []
    for it in items:
        source_type = it.get("source_type")
        behavior = it.get("behavior")
        log = {"id": it.get("id"), "status": ""}

        try:
            if source_type == "regulasi":
                chunk_id = (it.get("expected_chunk_ids") or [None])[0]
                chunk = find_regulasi_chunk(chunk_id, regulasi_dir) if chunk_id else None
                if not chunk:
                    print(f"  [SKIP] id={it['id']}: chunk regulasi '{chunk_id}' tidak ditemukan")
                    results.append({"id": it["id"], "status": "SKIP", "reason": "chunk regulasi tidak ditemukan"})
                    continue
                qa = gen_regulasi(chunk, log, model=model, target_behavior=behavior,
                                   reference_qa=(it["question"], it["ground_truth"]))
            elif source_type == "pengaduan":
                source_id = it.get("source_id_filter")
                c0 = find_pengaduan_chunk0(source_id, md_dir) if source_id else None
                if not c0:
                    print(f"  [SKIP] id={it['id']}: chunk pengaduan source_id={source_id} tidak ditemukan")
                    results.append({"id": it["id"], "status": "SKIP", "reason": "chunk pengaduan tidak ditemukan"})
                    continue

                extra_bodies = []
                for cid in it.get("expected_chunk_ids", []):
                    if cid == c0["chunk_id"]:
                        continue
                    extra_path = Path(md_dir) / f"{cid}.md"
                    if extra_path.exists():
                        parsed = parse_md_file(str(extra_path))
                        if parsed:
                            extra_bodies.append(parsed["body"])
                extra_context = "\n\n".join(extra_bodies) if extra_bodies else None

                qa = gen_pengaduan(source_id, {"0": c0}, log, model=model, target_behavior=behavior,
                                    reference_qa=(it["question"], it["ground_truth"]),
                                    extra_context=extra_context)
            else:
                continue

            if not qa:
                results.append({"id": it["id"], "status": "SKIP", "reason": "gen_pengaduan mengembalikan None"})
                continue

            results.append({
                "id": it["id"],
                "behavior": behavior,
                "source_type": source_type,
                "question_final": it["question"],
                "question_generated": qa["question"],
                "ground_truth_final": it["ground_truth"],
                "ground_truth_generated": qa["ground_truth"],
                "status": "OK",
            })
            print(f"  [OK] id={it['id']} {behavior}")
        except Exception as e:
            results.append({"id": it["id"], "status": "ERROR", "error": str(e)})
            print(f"  [ERROR] id={it['id']}: {e}")

        time.sleep(delay)

    if only_ids and os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            old_results = json.load(f)
        by_id = {r["id"]: r for r in old_results}
        for r in results:
            by_id[r["id"]] = r
        results = sorted(by_id.values(), key=lambda r: r["id"])

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    ok_count = sum(1 for r in results if r["status"] == "OK")
    print(f"\nSelesai: {ok_count}/{len(results)} berhasil digenerate ulang"
          f"{' (setelah digabung dengan laporan lama)' if only_ids else ''}.")
    print(f"Laporan tersimpan di: {out_path}")
    print(f"({dataset_path} TIDAK diubah -- ini murni laporan perbandingan untuk Anda nilai manual.)")
    return results


_NOT_FOUND_FALLBACK_TEMPLATES = [
    {
        "source_type": "pengaduan",
        "question": "Bagaimana prosedur pengaduan pelayanan publik di Kota Bandung?",
        "ground_truth": "Informasi ini tidak tersedia dalam korpus. Korpus ini khusus mencakup regulasi, SOP, dan data aduan Pemerintah Kabupaten Bandung, bukan Kota Bandung.",
        "expected_keyword": ["tidak tersedia", "Kabupaten Bandung"],
        "behavior": "not_found",
    },
    {
        "source_type": "pengaduan",
        "question": "Bagaimana status pengaduan dengan nomor tracking 9999999 tentang jalan berlubang di Kecamatan Cileunyi?",
        "ground_truth": "Nomor tracking 9999999 tidak ditemukan dalam basis pengetahuan. Silakan cek langsung melalui aplikasi LAPOR! atau hubungi instansi terkait.",
        "expected_keyword": ["tidak ditemukan", "nomor tracking"],
        "behavior": "not_found",
    },
    {
        "source_type": "regulasi",
        "question": "Berapa besar denda jika Perangkat Daerah terlambat menindaklanjuti pengaduan masyarakat?",
        "ground_truth": "Informasi mengenai nominal denda tidak tersedia dalam korpus regulasi yang ada. Regulasi yang tersedia tidak mengatur sanksi berupa denda finansial.",
        "expected_keyword": ["tidak tersedia", "denda"],
        "behavior": "not_found",
    },
]


def _verify_not_found_claim(qa, reg_items, peng_grouped):
    """Cek balik klaim 'tidak ada' dari LLM terhadap korpus asli -- KHUSUS untuk
    not_found, risikonya kebalik dari tipe soal lain: bukan LLM ngarang fakta yang
    kelihatan ada, tapi LLM bisa saja klaim 'X tidak ada' padahal X SEBENARNYA ada
    (mis. sebut 'Pasal 60' padahal pasal itu beneran eksis). Ground_truth yang lolos
    tanpa cek ini bisa jadi acuan evaluasi yang salah. Ini cuma best-effort: hanya
    bisa memverifikasi klaim yang textually checkable (nomor pasal spesifik, nomor
    tracking pengaduan spesifik) -- klaim jenis "wilayah/layanan di luar cakupan"
    tidak diverifikasi otomatis di sini (butuh penalaran domain, bukan pencarian
    teks), tetap mengandalkan kepatuhan LLM ke instruksi + review manual anda."""
    question = qa.get("question", "")

    for m in re.finditer(r"\bPasal\s+(\d+)\b", question, re.I):
        pasal_text = f"Pasal {m.group(1)}"
        if reg_items and any(pasal_text.lower() in it["body"].lower() for it in reg_items):
            return False, f"'{pasal_text}' ternyata ADA di korpus regulasi (klaim tidak valid)"

    for m in re.finditer(r"\b(\d{6,8})\b", question):
        num = m.group(1)
        if peng_grouped and num in peng_grouped:
            return False, f"nomor '{num}' ternyata ADA sbg source_id pengaduan asli (klaim tidak valid)"

    return True, None


def gen_not_found(log, model="gpt-4o", reg_items=None, peng_grouped=None):
    """Generate soal not_found via LLM (bukan template tetap) -- lihat SYSTEM_NOT_FOUND
    untuk kriteria lengkap (sesuai definisi 'not_found' di dokumentasi ringkasan
    statistik: jawaban tidak ada di korpus, sistem harus menolak dgn arahan yang
    membantu, bukan mengarang). Retry sampai 3x kalau verifikasi gagal (lihat
    _verify_not_found_claim); kalau tetap gagal, jatuh ke template lama sebagai
    jaring pengaman supaya pipeline tidak berhenti."""
    prompt = ("Buat 1 soal evaluasi kategori not_found sesuai kriteria di atas. "
              "Pilih salah satu pola (wilayah/layanan di luar cakupan/referensi tidak "
              "eksis/nomor tracking fiktif) secara bebas dan bervariasi.")
    for attempt in range(1, 4):
        try:
            raw = call_llm(prompt, SYSTEM_NOT_FOUND, temperature=0.9, model=model)
            qa = safe_parse_json(raw)
            if not qa.get("question") or not qa.get("ground_truth"):
                raise ValueError("question/ground_truth kosong dari LLM")
        except Exception as e:
            log.setdefault("rejected_attempts", []).append({"attempt": attempt, "reason": str(e)})
            continue

        ok, reason = _verify_not_found_claim(qa, reg_items, peng_grouped)
        if ok:
            qa["behavior"] = "not_found"
            log["raw_output"] = raw
            log["behavior"] = "not_found"
            log["verify_attempt"] = attempt
            return qa
        log.setdefault("rejected_attempts", []).append(
            {"attempt": attempt, "reason": reason, "question": qa.get("question")}
        )

    qa = dict(random.choice(_NOT_FOUND_FALLBACK_TEMPLATES))
    log["raw_output"] = "fallback_template (3x percobaan LLM gagal, lihat rejected_attempts)"
    log["behavior"] = "not_found"
    return qa


_KAMUS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kamus_variasi.json")
_KAMUS_CACHE = None


def load_kamus_variasi():
    """Muat kamus_variasi.json (singkatan/gaul/typo/sunda). Jika file tidak ada,
    kembalikan dict kosong -- generator tetap jalan (fallback ke contoh inline)."""
    global _KAMUS_CACHE
    if _KAMUS_CACHE is None:
        try:
            with open(_KAMUS_PATH, encoding="utf-8") as f:
                _KAMUS_CACHE = json.load(f)
        except Exception:
            _KAMUS_CACHE = {}
    return _KAMUS_CACHE


def build_kamus_reference(max_per_cat=12):
    """Rangkai contoh istilah dari kamus jadi teks referensi untuk prompt variasi,
    supaya variasi bahasa terkendali (bisa dipertanggungjawabkan), bukan asal LLM."""
    kamus = load_kamus_variasi()
    lines = []
    for cat in ("tidak_baku", "partikel_informal", "typo", "sunda"):
        d = kamus.get(cat, {})
        if not isinstance(d, dict):
            continue
        pairs = list(d.items())[:max_per_cat]
        contoh = "; ".join(
            f"{k}->{'/'.join(v) if isinstance(v, list) else v}" for k, v in pairs
        )
        if contoh:
            lines.append(f"- {cat}: {contoh}")
    return "\n".join(lines)


def gen_variasi(question, log, model="claude-sonnet-4-6"):
    ref = build_kamus_reference()
    ref_block = (
        f"\n\nGunakan referensi istilah dari kamus berikut bila cocok "
        f"(boleh menambah istilah lain yang wajar):\n{ref}" if ref else ""
    )
    prompt = f'Pertanyaan formal:\n"{question}"\n\nBuat 3 variasi gaya bahasa.{ref_block}'
    raw = call_llm(prompt, SYSTEM_VARIASI, temperature=0.4, model=model)
    variasi = safe_parse_json(raw)
    log["variasi_raw"] = raw
    return variasi


REQUIRED = ["kategori", "source_type", "jenis_filter", "source_id_filter",
            "reference_doc", "expected_chunk_ids", "question", "ground_truth",
            "expected_keyword", "behavior"]

NOT_FOUND_OPTIONAL_FIELDS = {"source_id_filter", "reference_doc"}

VALID_BEHAVIORS = {"answer_fact", "answer_list", "answer_procedure", "answer_temporal", "multi_hop", "not_found"}


LOW_VALUE_REGULASI_PATTERNS = [
    r"\bmenandatangani\b",
    r"\bditandatangani\b",
    r"tanda\s*tangan",
    r"\bnandatangan",
    r"\bntdg\b",
    r"\bditetapkan\s+di\b",
    r"\bditetapkan\s+pada\s+tanggal\b",
    r"\bmulai\s+berlaku\b",
    r"\bdiundangkan\b",
    r"\bbupati\s+yang\s+men",
    r"\bbupati\s+(?:mana|yang\s+mana|siapa)\b",
    r"\bsiapa\b.{0,20}\bbupati\b",
]


def is_low_value_regulasi_question(question):
    q = str(question or "").lower()
    return any(re.search(p, q) for p in LOW_VALUE_REGULASI_PATTERNS)


def validate(cand, valid_chunk_ids, existing_questions, dedup_threshold=DEDUP_THRESHOLD):
    errors = []
    is_not_found = cand.get("behavior") == "not_found"

    if cand.get("source_type") == "regulasi" and is_low_value_regulasi_question(cand.get("question", "")):
        errors.append(
            "pertanyaan low-value (tanggal penetapan/penandatangan) -- "
            "dilarang di SYSTEM_REGULASI kriteria #6"
        )
    for f in REQUIRED:
        if is_not_found and f in NOT_FOUND_OPTIONAL_FIELDS:
            continue
        if f not in cand or cand[f] is None:
            errors.append(f"field '{f}' kosong")

    for cid in cand.get("expected_chunk_ids", []):
        if cid and cid not in valid_chunk_ids:
            errors.append(f"chunk_id '{cid}' tidak ada di korpus")

    if cand.get("behavior") not in VALID_BEHAVIORS:
        errors.append(f"behavior '{cand.get('behavior')}' tidak valid")

    if len(cand.get("question", "")) < 10:
        errors.append("pertanyaan terlalu pendek")

    if len(cand.get("ground_truth", "")) < 10:
        errors.append("ground_truth terlalu pendek")

    dup_sim = 0.0
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        if existing_questions:
            corpus = existing_questions + [cand["question"]]
            vec = TfidfVectorizer().fit_transform(corpus)
            sims = cosine_similarity(vec[-1:], vec[:-1])[0]
            dup_sim = float(sims.max())
            if dup_sim >= dedup_threshold:
                errors.append(f"duplikat (sim={dup_sim:.2f})")
    except Exception:
        pass

    return errors, dup_sim


def run_pipeline(args):
    print("\n" + "="*60)
    print("PIPELINE PEMBANGUNAN GOLDEN DATASET")
    print("="*60)

    random.seed(args.seed)

    existing_questions_seed = []
    used_chunk_ids, used_source_ids = set(), set()
    if args.existing:
        print(f"\n[0/5] Memuat dataset existing dari {args.existing} ...")
        existing_items = [json.loads(l) for l in open(args.existing, encoding="utf-8")]
        for it in existing_items:
            used_chunk_ids.update(it.get("expected_chunk_ids", []))
            if it.get("source_id_filter"):
                used_source_ids.add(it["source_id_filter"])
            if it.get("question"):
                existing_questions_seed.append(it["question"])
        print(f"      {len(existing_items)} entri lama -- {len(used_chunk_ids)} chunk_id & "
              f"{len(used_source_ids)} source_id akan di-exclude")

    print(f"\n[1/5] Memuat korpus ...")
    reg_items, peng_grouped = load_corpus(args.corpus_regulasi_dir, args.corpus_md_dir)
    if used_chunk_ids:
        reg_items = [it for it in reg_items if it["chunk_id"] not in used_chunk_ids]
    if used_source_ids:
        peng_grouped = {sid: c for sid, c in peng_grouped.items() if sid not in used_source_ids}
    valid_chunk_ids = (
        {it["chunk_id"] for it in reg_items} |
        {c["chunk_id"] for chunks in peng_grouped.values() for c in chunks.values()}
    )
    print(f"      {len(reg_items)} chunk regulasi, {len(peng_grouped)} aduan unik (setelah exclude)")

    print(f"\n[2/5] Menerapkan filter pra-generasi ...")
    print(f"      Minimum panjang chunk: {MIN_CHUNK_LENGTH} karakter")
    print(f"      Filter boilerplate: aktif")
    print(f"      Threshold deduplikasi: {DEDUP_THRESHOLD}")

    n_total = args.n
    n_not_found = max(1, round(n_total * 0.05))
    n_var_base = args.n_variasi_basis
    n_var_soal = n_var_base * 3

    n_utama = n_total - n_not_found - n_var_soal
    if n_utama < 0:
        print(f"[WARN] n_variasi_basis={n_var_base} (={n_var_soal} soal) terlalu besar untuk n={n_total}. Kurangi n_variasi_basis.")
        n_var_base = max(0, (n_total - n_not_found) // 4)
        n_var_soal = n_var_base * 3
        n_utama = n_total - n_not_found - n_var_soal

    if args.source_type == "regulasi":
        n_reg = n_utama; n_peng = 0
    elif args.source_type == "pengaduan":
        n_reg = 0; n_peng = n_utama
    else:
        n_peng = round(n_utama * 0.60)
        n_reg = n_utama - n_peng

    def make_behavior_schedule(n, dist):
        """Hasilkan list behavior sesuai distribusi target, dikocok acak."""
        schedule = []
        for beh, ratio in dist.items():
            count = max(1, round(n * ratio)) if ratio > 0 else 0
            schedule.extend([beh] * count)
        while len(schedule) < n:
            schedule.append(max(dist, key=dist.get))
        schedule = schedule[:n]
        random.shuffle(schedule)
        return schedule

    reg_schedule = make_behavior_schedule(n_reg, TARGET_DIST["regulasi"]) if n_reg > 0 else []
    peng_schedule = make_behavior_schedule(n_peng, TARGET_DIST["pengaduan"]) if n_peng > 0 else []

    print(f"\n[3/5] Target: {n_reg} regulasi, {n_peng} pengaduan, {n_not_found} not_found, {n_var_soal} variasi")
    print(f"      Total output: {n_reg + n_peng + n_not_found + n_var_soal} soal")
    if n_reg > 0:
        print(f"      Distribusi regulasi: {dict(Counter(reg_schedule))}")
    if n_peng > 0:
        print(f"      Distribusi pengaduan: {dict(Counter(peng_schedule))}")

    reg_pool = list(reg_items)
    random.shuffle(reg_pool)
    peng_pool = [
        sid for sid in peng_grouped
        if has_substantive_tindak_lanjut((peng_grouped[sid].get("0") or peng_grouped[sid].get(0) or {}))
    ]
    print(f"      Pool pengaduan substantif: {len(peng_pool)} dari {len(peng_grouped)} aduan")
    random.shuffle(peng_pool)

    candidates = []
    existing_questions = list(existing_questions_seed)
    log_entries = []
    new_id = 1

    print(f"\n[4/5] Generasi soal ...")
    reg_success = 0
    for chunk in reg_pool:
        if reg_success >= n_reg:
            break
        target_beh = reg_schedule[reg_success] if reg_success < len(reg_schedule) else "answer_fact"
        qa, actual_beh, attempt_logs = _gen_regulasi_with_fallback(chunk, target_beh, args.model)
        log_entries.extend(attempt_logs)
        if qa is not None:
            meta = chunk["meta"]
            cand = {
                "id": new_id,
                "kategori": "Regulasi",
                "source_type": "regulasi",
                "jenis_filter": "regulasi",
                "source_id_filter": meta.get("source_id", ""),
                "reference_doc": chunk["chunk_id"],
                "expected_chunk_ids": [chunk["chunk_id"]],
                "expected_source_ids": [meta["source_id"]] if meta.get("source_id") else [],
                "question": qa["question"],
                "ground_truth": qa["ground_truth"],
                "expected_keyword": qa.get("expected_keyword", []),
                "behavior": actual_beh,
            }
            errors, sim = validate(cand, valid_chunk_ids, existing_questions)
            last_log = attempt_logs[-1]
            if errors:
                last_log["status"] = "SKIP"; last_log["errors"] = errors
                print(f"  [SKIP] regulasi {chunk['chunk_id'][:35]}: {errors[0]}")
            else:
                candidates.append(cand)
                existing_questions.append(cand["question"])
                last_log["status"] = "OK"; last_log["id"] = new_id; last_log["dup_sim"] = sim
                print(f"  [OK] id={new_id} {actual_beh:15s} | {cand['question'][:55]}")
                new_id += 1; reg_success += 1
        time.sleep(args.delay)

    peng_success = 0
    for sid in peng_pool:
        if peng_success >= n_peng:
            break
        target_beh = peng_schedule[peng_success] if peng_success < len(peng_schedule) else "answer_fact"
        chunks = peng_grouped[sid]
        qa, actual_beh, attempt_logs = _gen_pengaduan_with_fallback(sid, chunks, target_beh, args.model)
        log_entries.extend(attempt_logs)
        if qa is not None:
            c0 = chunks.get("0") or chunks.get(0)
            cand = {
                "id": new_id,
                "kategori": "Pengaduan",
                "source_type": "pengaduan",
                "jenis_filter": "pengaduan",
                "source_id_filter": sid,
                "reference_doc": c0["chunk_id"],
                "expected_chunk_ids": [c0["chunk_id"]],
                "expected_source_ids": [sid],
                "question": qa["question"],
                "ground_truth": qa["ground_truth"],
                "expected_keyword": qa.get("expected_keyword", []),
                "behavior": actual_beh,
            }
            errors, sim = validate(cand, valid_chunk_ids, existing_questions)
            last_log = attempt_logs[-1]
            if errors:
                last_log["status"] = "SKIP"; last_log["errors"] = errors
                print(f"  [SKIP] pengaduan {sid}: {errors[0]}")
            else:
                candidates.append(cand)
                existing_questions.append(cand["question"])
                last_log["status"] = "OK"; last_log["id"] = new_id; last_log["dup_sim"] = sim
                print(f"  [OK] id={new_id} {actual_beh:15s} | {cand['question'][:55]}")
                new_id += 1; peng_success += 1
        time.sleep(args.delay)

    not_found_success = 0
    not_found_attempts = 0
    max_not_found_attempts = max(n_not_found * 4, 8)
    while not_found_success < n_not_found and not_found_attempts < max_not_found_attempts:
        not_found_attempts += 1
        log = {"step": "not_found", "status": ""}
        qa = gen_not_found(log, model=args.model, reg_items=reg_items, peng_grouped=peng_grouped)
        src_type = qa.get("source_type", "regulasi")
        cand = {
            "id": new_id,
            "kategori": "Out of Scope",
            "source_type": src_type,
            "jenis_filter": src_type,
            "source_id_filter": None,
            "reference_doc": None,
            "expected_chunk_ids": [],
            "expected_source_ids": [],
            "question": qa["question"],
            "ground_truth": qa["ground_truth"],
            "expected_keyword": qa.get("expected_keyword", []),
            "behavior": "not_found",
        }
        errors, sim = validate(cand, valid_chunk_ids, existing_questions)
        if errors:
            log["status"] = "SKIP"; log["errors"] = errors
            print(f"  [SKIP] not_found: {errors[0]}")
            log_entries.append(log)
            continue
        candidates.append(cand)
        existing_questions.append(cand["question"])
        log["status"] = "OK"; log["id"] = new_id
        print(f"  [OK] id={new_id} not_found        | {cand['question'][:55]}")
        new_id += 1
        not_found_success += 1
        log_entries.append(log)

    if not_found_success < n_not_found:
        print(f"  [WARN] not_found cuma {not_found_success}/{n_not_found} "
              f"(mentok di {max_not_found_attempts}x percobaan) -- soal lain di "
              f"korpus tidak terpengaruh, cuma kategori ini kurang dari target.")

    if n_var_base > 0:
        _fact = [c for c in candidates if c["behavior"] == "answer_fact"]
        _peng_fact = [c for c in _fact if c.get("source_type") == "pengaduan"]
        _reg_fact = [c for c in _fact if c.get("source_type") == "regulasi"]
        basis_pool = []
        _pi = _ri = 0
        while _pi < len(_peng_fact) or _ri < len(_reg_fact):
            for _ in range(3):
                if _pi < len(_peng_fact):
                    basis_pool.append(_peng_fact[_pi]); _pi += 1
            for _ in range(2):
                if _ri < len(_reg_fact):
                    basis_pool.append(_reg_fact[_ri]); _ri += 1
        variasi_all = []
        basis_idx = 0
        basis_used = 0
        max_basis_attempts = max(n_var_base * 3, 10)
        print(f"\n      Variasi bahasa (target {n_var_soal} soal dari basis) ...")
        while (len(variasi_all) < n_var_soal
               and basis_idx < len(basis_pool)
               and basis_used < max_basis_attempts):
            base = basis_pool[basis_idx]
            basis_idx += 1
            basis_used += 1
            log = {"step": "variasi", "base_id": base["id"], "status": ""}
            try:
                vars_ = gen_variasi(base["question"], log, model=args.model)
                for var in vars_:
                    if len(variasi_all) >= n_var_soal:
                        break
                    var_cand = dict(base)
                    var_cand["kategori"] = base["kategori"] + " (Variasi Bahasa)"
                    var_cand["question"] = var["question"]
                    var_cand["_gaya_bahasa"] = var["gaya"]
                    var_cand["_variasi_dari_id"] = base["id"]
                    var_errors, var_sim = validate(var_cand, valid_chunk_ids, existing_questions)
                    if var_errors:
                        print(f"  [SKIP] variasi base_id={base['id']} gaya={var['gaya']}: {var_errors[0]}")
                        continue
                    var_cand["id"] = new_id
                    variasi_all.append(var_cand)
                    existing_questions.append(var_cand["question"])
                    print(f"  [OK] id={new_id} variasi_{var['gaya']:12s} | {var['question'][:50]}")
                    new_id += 1
                log["status"] = "OK"
            except Exception as e:
                log["status"] = "ERROR"; log["error"] = str(e)
                print(f"  [ERROR] variasi id={base['id']}: {e}")
            log_entries.append(log)
            time.sleep(args.delay)
        if len(variasi_all) < n_var_soal:
            print(f"  [WARN] variasi cuma {len(variasi_all)}/{n_var_soal} "
                  f"(basis pool habis atau mentok {max_basis_attempts}x percobaan).")
        candidates += variasi_all

    print(f"\n[5/5] Menyimpan hasil ...")
    with open(args.output, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    log_path = args.output.replace(".jsonl", "_log.json")
    log_data = {
        "config": vars(args),
        "summary": {
            "total_kandidat": len(candidates),
            "regulasi": sum(1 for c in candidates if c["source_type"] == "regulasi" and not c.get("_gaya_bahasa")),
            "pengaduan": sum(1 for c in candidates if c["source_type"] == "pengaduan" and not c.get("_gaya_bahasa")),
            "not_found": sum(1 for c in candidates if c["behavior"] == "not_found"),
            "variasi_bahasa": sum(1 for c in candidates if c.get("_gaya_bahasa")),
            "distribusi_behavior": dict(Counter(c["behavior"] for c in candidates)),
        },
        "langkah_validasi": {
            "filter_panjang_minimum": f"{MIN_CHUNK_LENGTH} karakter",
            "filter_boilerplate": "aktif",
            "validasi_skema": "field wajib + chunk_id di korpus",
            "deduplikasi_semantik": f"TF-IDF cosine ≥ {DEDUP_THRESHOLD}",
            "RTC_check": "PENDING — jalankan evaluate_mrr.py setelah ini",
        },
        "log_detail": log_entries,
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"SELESAI")
    print(f"{'='*60}")
    print(f"Dataset  : {args.output} ({len(candidates)} soal)")
    print(f"Log      : {log_path}")
    dist = Counter(c["behavior"] for c in candidates)
    for beh, n in sorted(dist.items()):
        print(f"  {beh:20s}: {n}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n", type=int, default=100,
                        help="Total soal yang di-generate (default: 100)")
    parser.add_argument("--source-type",
                        choices=["regulasi", "pengaduan", "both"],
                        default="both",
                        help="Sumber dokumen (default: both)")
    parser.add_argument("--n-variasi-basis", type=int, default=8,
                        help="Jumlah soal basis untuk variasi bahasa (default: 8, 0=matikan)")
    parser.add_argument("--corpus-md-dir", default="corpus_aduan",
                        help="Korpus pengaduan aktif (default: corpus_aduan; corpus_md lama sudah diarsipkan)")
    parser.add_argument("--corpus-regulasi-dir", default="corpus_regulasi_sop",
                        help="Korpus regulasi aktif (default: corpus_regulasi_sop; corpus_regulasi lama sudah diarsipkan)")
    parser.add_argument("--existing", default=None,
                        help="Golden dataset yang sudah ada (jsonl) -- kalau diisi, exclude source_id/chunk_id yang "
                             "sudah dipakai dan pra-isi dedup check dengan pertanyaan yang sudah ada")
    parser.add_argument("--output", default="golden_dataset_output.jsonl")
    parser.add_argument("--seed", type=int, default=2024,
                        help="Random seed untuk reprodusibilitas (default: 2024)")
    parser.add_argument("--model",
                        default="gpt-4o",
                        help="Model generator (default: gpt-4o, butuh OPENAI_API_KEY). "
                             "Pilihan Anthropic (kalau kredit Anthropic tersedia lagi): "
                             "claude-haiku-4-5-20251001 | claude-sonnet-4-6 (butuh ANTHROPIC_API_KEY) "
                             "-- nama model diawali 'gpt' otomatis dirutekan ke OpenAI, selain itu ke Anthropic.")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Jeda antar API call dalam detik (default: 0.5)")
    parser.add_argument("--validate-existing", default=None,
                        help="Mode backfill/maintenance (TIDAK memanggil Claude API, TIDAK mengubah "
                             "isi): validasi ulang golden dataset yang SUDAH final terhadap korpus "
                             "produksi terkini (schema + keberadaan expected_chunk_ids). "
                             "Contoh: --validate-existing golden_dataset_fix.jsonl")
    parser.add_argument("--consistency-check", default=None,
                        help="Mode pembanding semantik (MEMANGGIL Claude API, 1 call per item, TIDAK "
                             "mengubah dataset asli): generate ulang tiap item dari chunk sumber yang "
                             "sama, simpan berdampingan dengan versi final untuk dinilai manual. "
                             "Contoh: --consistency-check golden_dataset_fix.jsonl")
    parser.add_argument("--consistency-limit", type=int, default=None,
                        help="Batasi jumlah item yang diproses --consistency-check (default: semua)")
    parser.add_argument("--consistency-out", default=None,
                        help="Path file laporan --consistency-check (default: <dataset>_consistency_report.json)")
    parser.add_argument("--consistency-ids", default=None,
                        help="Batasi --consistency-check ke id tertentu saja (pisah koma, mis. '4,52,61'). "
                             "Hasil DIGABUNG ke laporan lama (kalau ada), bukan menimpa semuanya -- cocok "
                             "buat retry beberapa item saja (mis. pakai --model lain) tanpa run ulang semua.")
    args = parser.parse_args()

    if args.validate_existing:
        items, report = validate_existing_dataset(
            args.validate_existing, args.corpus_regulasi_dir, args.corpus_md_dir
        )
        print_validation_report(report, len(items))
        sys.exit(0 if not report["missing_chunk"] and not report["missing_field"] else 1)

    if args.consistency_check:
        key_name = required_api_key_name(args.model)
        if not os.getenv(key_name):
            print(f"[ERROR] {key_name} belum di-set (dibutuhkan untuk --model {args.model}).", file=sys.stderr)
            sys.exit(1)
        out_path = args.consistency_out or args.consistency_check.replace(".jsonl", "_consistency_report.json")
        only_ids = None
        if args.consistency_ids:
            only_ids = {int(x.strip()) for x in args.consistency_ids.split(",") if x.strip()}
        run_consistency_check(
            args.consistency_check, args.corpus_regulasi_dir, args.corpus_md_dir,
            model=args.model, delay=args.delay, limit=args.consistency_limit, out_path=out_path,
            only_ids=only_ids,
        )
        sys.exit(0)

    key_name = required_api_key_name(args.model)
    if not os.getenv(key_name):
        print(f"[ERROR] {key_name} belum di-set (dibutuhkan untuk --model {args.model}).", file=sys.stderr)
        sys.exit(1)

    run_pipeline(args)


if __name__ == "__main__":
    main()