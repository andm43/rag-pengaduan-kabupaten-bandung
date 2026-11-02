# RAG Pengaduan Masyarakat Kabupaten Bandung (SP4N-LAPOR!)

Kode program dan data skripsi **"Implementasi dan Evaluasi Kinerja Sistem Tanya Jawab Berbasis Retrieval-Augmented Generation Menggunakan Model Qwen 2.5 pada Data Pengaduan Masyarakat Kabupaten Bandung"**.
Andre Mahesa Somantri (10122405), Teknik Informatika UNIKOM.

Komponen utama:

- Generator: `qwen2.5:7b` melalui Ollama (penelitian menjalankan Ollama di RunPod)
- Embedding: `intfloat/multilingual-e5-base`
- Vector store: ChromaDB plus indeks BM25 (hybrid retrieval)
- Antarmuka: bot Telegram dan dashboard admin Streamlit
- Evaluasi: MRR (retrieval, evaluasi utama) dan RAGAS (kualitas jawaban, pendukung)

## Struktur folder

Skrip membaca path relatif terhadap folder ini (`BASE_DIR`). Karena itu file inti tetap di root.

```
.
├── preprocessing.py          1. Gabung & bersihkan dataset/*.xlsx -> dataset_clean.csv
├── anonymize.py                 (modul) anonimisasi data pribadi, dipakai build_corpus_aduan.py
├── build_corpus_aduan.py     2. dataset_clean.csv -> corpus_aduan/
├── build_corpus_docling.py   3. PDF regulasi di pendukung/ -> corpus_regulasi_sop/
├── build_corpus_sop.py       4. PDF SOP di pendukung/ -> corpus_regulasi_sop/SOP_Pengaduan_*.md
├── build_chroma_index.py     5. Korpus -> chroma_rag/ (ChromaDB)
├── build_bm25_index.py       6. Korpus -> chroma_rag/bm25_index.pkl
├── ask_rag.py                   Inti RAG: retrieval, reranking, prompt, guardrail, generasi
├── gen_golden_dataset.py        Pembangkit draf golden dataset (sebelum verifikasi manual)
├── build_kamus_data.py          Chunk "kamus data" (eksperimen, TIDAK masuk indeks hasil akhir)
├── evaluate_mrr.py              Evaluasi retrieval (MRR, Hit@k)
├── evaluate_ragas.py            Evaluasi jawaban (RAGAS)
├── telegram_bot.py              Bot Telegram
├── db.py                        Penyimpanan riwayat chat bot (telegram_bot.db, SQLite)
├── admin_dashboard.py           Dashboard admin Streamlit (jalankan_dashboard.bat)
├── pipeline_manager.py          (modul dashboard) unggah dataset/regulasi baru & bangun ulang indeks
├── search_corpus.py             (modul dashboard) pencarian teks di korpus
├── extract_pdf.py               (modul dashboard) ekstraksi teks PDF regulasi yang diunggah
│
├── dataset/                     Data mentah SP4N-LAPOR! 2020-2025 (.xlsx)       [tidak di repo]
├── dataset_clean.csv            Hasil preprocessing                             [tidak di repo]
├── pendukung/                   PDF regulasi dan SOP sumber korpus
├── corpus_aduan/                Korpus pengaduan (markdown per chunk)           [tidak di repo]
├── corpus_regulasi_sop/         Korpus regulasi + SOP (markdown per chunk)
├── chroma_rag/                  Indeks ChromaDB + bm25_index.pkl hasil akhir    [tidak di repo]
├── kamus_variasi.json           Kamus variasi bahasa untuk gen_golden_dataset.py
├── golden_dataset_fix.jsonl     Golden dataset final (100 soal)
├── mrr_result_final.json        Hasil MRR final
├── ragas_results_100_7B_rag.json    Hasil RAGAS final (RAG), lengkap              [tidak di repo]
├── ragas_results_100_7B_norag.json  Hasil RAGAS baseline (tanpa RAG), lengkap      [tidak di repo]
├── ragas_ringkasan_100_7B_*.json    Ringkasan kedua file di atas: skor per soal, tanpa teks aduan
├── regulasi_uploads/            Tempat unggahan PDF dari dashboard
├── .env.example                 Contoh isi .env
└── LICENSE                      MIT
```

## Hasil final

Sumber angka: `mrr_result_final.json` dan `ragas_ringkasan_100_7B_rag.json`/`_norag.json`.

| Metrik | Nilai |
|---|---|
| MRR (n=95 soal terjawab) | 0,9246 |
| Hit@5 | 93/95 (97,9%) |
| Faithfulness (n=92) | 0,8308 |
| Answer Relevancy (n=92) | 0,7009 (No-RAG 0,7357) |
| Context Precision (n=92) | 0,7544 |
| Context Recall (n=95) | 0,8742 |
| Semantic Similarity (n=95) | 0,8702 (No-RAG 0,8563) |

Golden dataset berisi 100 soal: 95 bisa dijawab, 5 di luar cakupan korpus. Guardrail non-generatif menjawab id 8, 38, dan 54, jadi Faithfulness, Answer Relevancy, dan Context Precision tidak menghitung ketiganya.

## Data

Repo ini tidak memuat data pengaduan. Data mentah SP4N-LAPOR! Kabupaten Bandung berisi nama, nomor telepon, NIK, dan alamat pelapor, dan penggunaannya terikat izin dari instansi. Anonimisasi di `anonymize.py` mengganti NIK dan nomor telepon, tetapi tidak menjamin semua nama dan alamat di teks bebas tersamarkan.

Yang tersedia di repo:

- `pendukung/` dan `corpus_regulasi_sop/`: regulasi dan SOP publik
- `golden_dataset_fix.jsonl`, `mrr_result_final.json`, dan `ragas_ringkasan_*.json`, untuk memeriksa angka skripsi. File RAGAS lengkap tidak disertakan karena memuat konteks aduan hasil retrieval.

Untuk menjalankan ulang pipeline, siapkan `dataset/dataset_<tahun>.xlsx` hasil ekspor portal SP4N-LAPOR! dengan kolom yang dibaca `preprocessing.py`. `dataset_clean.csv` hasil preprocessing berisi kolom:

```
id, tahun, judul, isi_laporan, tindak_lanjut, instansi, kecamatan, kategori,
klasifikasi, status, tanggal_masuk, tanggal_tindak_lanjut_pertama,
tanggal_selesai, durasi_respon, durasi_penyelesaian
```

## Prasyarat

- Python 3.10
- Ollama dengan model `qwen2.5:7b` (`ollama pull qwen2.5:7b`)
- GPU CUDA wajib untuk `build_chroma_index.py` (skrip menolak jalan tanpa CUDA). `ask_rag.py` memakai CUDA bila tersedia
- Tesseract OCR untuk `build_corpus_docling.py` (path di konstanta `TESSERACT_CMD`)
- File `.env` di root (salin dari `.env.example`) dengan variabel berikut:
  - `OPENAI_API_KEY` (judge RAGAS `gpt-4o`, pembangkit golden dataset)
  - `ANTHROPIC_API_KEY` (opsional)
  - `TELEGRAM_BOT_TOKEN`, `TELEGRAM_ADMIN_GROUP_ID`, `TELEGRAM_MODEL_NAME`, `PHOTOS_DIR` (bot)
  - `OLLAMA_URL` (default `http://localhost:11434`)

## Instalasi

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu121   # sesuaikan dengan perangkat
pip install -r requirements.txt
```

`requirements.txt` tidak memuat nomor versi. Penelitian tidak mencatat versi paket.

## Menjalankan dari awal

Langkah ini butuh data mentah di `dataset/` (lihat bagian Data). Indeks hasil akhir berisi 11.475 chunk: 11.232 pengaduan dan 243 regulasi/SOP.

```powershell
python preprocessing.py
python build_corpus_aduan.py
python build_corpus_docling.py
python build_corpus_sop.py
python build_chroma_index.py
python build_bm25_index.py
```

Jalankan `build_corpus_docling.py` sebelum `build_corpus_sop.py`. Keduanya menulis ke `corpus_regulasi_sop/`, dan skrip docling mengosongkan folder itu lebih dulu (kecuali file `SOP_Pengaduan_*`).

## Evaluasi

Perintah di bawah menghasilkan angka skripsi (retrieval kondisi produksi, tanpa pembatasan sumber dokumen):

```powershell
# MRR
python evaluate_mrr.py --no-source-scope --output mrr_result_final.json

# RAGAS dengan RAG
python evaluate_ragas.py --no-source-scope --context-metrics --judge-provider openai --evaluator-model gpt-4o --ollama-url http://<host>:<port>/api/generate --output ragas_results_100_7B_rag.json

# RAGAS baseline tanpa RAG
python evaluate_ragas.py --no-rag --judge-provider openai --evaluator-model gpt-4o --ollama-url http://<host>:<port>/api/generate --output ragas_results_100_7B_norag.json
```

Catatan:

- Pakai nama `--output` lain agar file hasil final tidak tertimpa.
- Kode mengunci seed retrieval, generator, dan judge. Hasil sama bila versi Ollama, model, dan perangkat sama. Judge RAGAS berupa LLM eksternal, jadi nilai bisa bergeser tipis antar-run.
- Untuk uji cepat, tambahkan `--max-samples 10`.

## Antarmuka

```powershell
python telegram_bot.py        # bot Telegram, lihat TELEGRAM_SETUP.md
jalankan_dashboard.bat        # dashboard admin di http://localhost:8501
```

## Lisensi

Kode dirilis dengan lisensi MIT (lihat `LICENSE`). Lisensi ini tidak berlaku untuk data pengaduan.
