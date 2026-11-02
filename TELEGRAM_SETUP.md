# Telegram Trial

File ini menyiapkan bot Telegram lokal di atas engine `ask_rag.py`.

## 1. Buat bot Telegram

1. Buka `@BotFather` di Telegram.
2. Jalankan `/newbot`.
3. Simpan token bot yang diberikan.

## 2. Siapkan environment variable

Cara yang dipakai sekarang: taruh di file `.env` di root project (sudah
otomatis dibaca lewat `load_dotenv()` di `telegram_bot.py`, jadi tidak perlu set `$env:` manual tiap sesi
PowerShell).

> **PENTING:** `OLLAMA_URL` (Subbab 4b di bawah) **belum** ada di `.env` saat ini.
> URL proxy RunPod berubah setiap kali pod baru dinyalakan, sehingga tidak aman
> disimpan statis di `.env`. Ini wajib di-set manual lewat `$env:OLLAMA_URL=...`
> di setiap sesi PowerShell baru sebelum menjalankan bot -- kalau
> lupa, `ask_rag.py` diam-diam fallback ke `http://localhost:11434` dan bot akan
> gagal generate jawaban tanpa pesan error yang jelas.

`.env`:

```
TELEGRAM_BOT_TOKEN=isi-token-dari-BotFather
TELEGRAM_MODEL_NAME=qwen2.5:7b
TELEGRAM_TOP_K=5
TELEGRAM_TEMPERATURE=0.1
```

Opsional (tambahkan di `.env` juga kalau perlu):

```
TELEGRAM_LONG_POLL_TIMEOUT=30
TELEGRAM_MAX_HISTORY=12
TELEGRAM_ANSWER_TIMEOUT=45
TELEGRAM_ADMIN_GROUP_ID=
```

- `TELEGRAM_ANSWER_TIMEOUT`: batas waktu (detik) tunggu jawaban dari pipeline RAG sebelum bot menyerah dan memberi tahu pengguna.
- `TELEGRAM_ADMIN_GROUP_ID`: chat ID grup admin yang menerima notifikasi saat bot menandai percakapan butuh operator handoff.

Alternatif tanpa `.env` (set langsung di sesi PowerShell, tidak persisten):

```powershell
$env:TELEGRAM_BOT_TOKEN="isi-token-dari-BotFather"
$env:TELEGRAM_MODEL_NAME="qwen2.5:7b"
```

## 3. Bangun ulang corpus dan index

Jalankan ulang jika corpus atau metadata baru saja berubah:

```powershell
python build_corpus_aduan.py
python build_corpus_docling.py
python build_corpus_sop.py
python build_chroma_index.py
python build_bm25_index.py
```

## 4. Pastikan Ollama aktif

Bot ini memakai `ask_rag.py`, jadi service Ollama harus hidup dan bisa dijangkau.

### 4a. Ollama lokal (default)

```powershell
ollama serve
```

Dan model yang dipakai harus tersedia, misalnya:

```powershell
ollama list
```

### 4b. Ollama di RunPod (skenario split: bot & data di laptop, GPU di RunPod)

Bot Telegram, riwayat chat (`telegram_bot.db`), dan index (`chroma_rag/`) tetap
jalan/tersimpan di laptop. Hanya generasi jawaban (`ask_rag.generate_answer`) yang
dikirim ke Ollama di pod RunPod.

1. Pastikan Ollama sudah jalan di pod dan port `11434` sudah di-expose sebagai HTTP port
   (cek dashboard RunPod → pod → tab **Connect**). URL proxy-nya berbentuk:
   `https://<POD_ID>-11434.proxy.runpod.net`
2. Set env var `OLLAMA_URL` ke URL itu (base URL saja, tanpa `/api/...`) sebelum
   menjalankan bot:

```powershell
$env:OLLAMA_URL="https://<POD_ID>-11434.proxy.runpod.net"
$env:OLLAMA_URL="http://<host>:<port>"
```

`ask_rag.py` otomatis menambahkan `/api/generate` ke base URL ini.

> Catatan keamanan: Ollama tidak punya autentikasi bawaan. Selama port itu ter-expose
> publik lewat proxy RunPod, siapa pun yang tahu URL-nya bisa memakai GPU-mu. Jangan
> sebarkan URL ini, dan matikan pod kalau sedang tidak dipakai.

## 5. Jalankan bot

```powershell
python telegram_bot.py
```

## 6. Uji dari Telegram

Kirim pesan ke bot:

- `Bagaimana alur disposisi pengaduan di SP4N-LAPOR?`
- `Jalan berlubang di Soreang harus ke dinas mana?`
- `Apa dasar hukum pengelolaan pengaduan?`

Perintah yang tersedia:

- `/start`
- `/help`
- `/reset`

## Catatan

- Bot ini memakai `long polling`, jadi belum butuh domain publik atau webhook.
- Hasil `disposisi` masih berupa `usulan instansi`, bukan keputusan final.
- Riwayat chat lokal disimpan di `telegram_bot.db` (SQLite).
