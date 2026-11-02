"""Lapisan persistensi SQLite untuk bot Telegram pengaduan.

Public API: load_state(), save_state(), log_complaint_row(), load_complaints().
telegram_bot.py dan admin_dashboard.py memanggil fungsi-fungsi ini
tanpa perlu tahu backend penyimpanan.

Catatan: migrasi ke PostgreSQL direkomendasikan untuk deployment
multi-proses di VPS (lihat Saran pada Bab 5).
"""
import json
import sqlite3
import threading
from pathlib import Path

DB_PATH = Path(__file__).parent / "telegram_bot.db"

_lock = threading.Lock()
_initialized = False


def get_connection():
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    global _initialized
    if _initialized:
        return
    conn = get_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                data TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS complaints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kode TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                timestamp_unix REAL NOT NULL,
                chat_id TEXT NOT NULL,
                masalah TEXT NOT NULL DEFAULT '',
                lokasi TEXT NOT NULL DEFAULT '',
                waktu_kejadian TEXT NOT NULL DEFAULT '',
                has_foto INTEGER NOT NULL DEFAULT 0,
                foto_path TEXT NOT NULL DEFAULT '',
                tahun INTEGER,
                kategori_perkiraan TEXT NOT NULL DEFAULT '',
                instansi_perkiraan TEXT NOT NULL DEFAULT '',
                kecamatan_perkiraan TEXT NOT NULL DEFAULT ''
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_complaints_kode ON complaints(kode)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_complaints_chat_id ON complaints(chat_id)"
        )
        conn.commit()
    finally:
        conn.close()
    _initialized = True


def load_state():
    init_db()
    conn = get_connection()
    try:
        row = conn.execute("SELECT data FROM state WHERE id = 1").fetchone()
    finally:
        conn.close()
    if row is None:
        return {"last_update_id": None, "chat_histories": {}}
    try:
        return json.loads(row["data"])
    except json.JSONDecodeError:
        return {"last_update_id": None, "chat_histories": {}}


def save_state(state):
    init_db()
    payload = json.dumps(state, ensure_ascii=False)
    with _lock:
        conn = get_connection()
        try:
            conn.execute(
                """
                INSERT INTO state (id, data) VALUES (1, ?)
                ON CONFLICT (id) DO UPDATE SET data = excluded.data
                """,
                (payload,),
            )
            conn.commit()
        finally:
            conn.close()


def log_complaint_row(entry: dict):
    """entry: kode, timestamp, timestamp_unix, chat_id, masalah, lokasi,
    waktu_kejadian, has_foto, foto_path, tahun, kategori_perkiraan,
    instansi_perkiraan, kecamatan_perkiraan."""
    init_db()
    with _lock:
        conn = get_connection()
        try:
            conn.execute(
                """
                INSERT INTO complaints
                (kode, timestamp, timestamp_unix, chat_id, masalah, lokasi,
                 waktu_kejadian, has_foto, foto_path, tahun,
                 kategori_perkiraan, instansi_perkiraan, kecamatan_perkiraan)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry["kode"],
                    entry["timestamp"],
                    entry["timestamp_unix"],
                    str(entry["chat_id"]),
                    entry.get("masalah", ""),
                    entry.get("lokasi", ""),
                    entry.get("waktu_kejadian", ""),
                    int(bool(entry.get("has_foto", False))),
                    entry.get("foto_path", ""),
                    entry.get("tahun"),
                    entry.get("kategori_perkiraan", ""),
                    entry.get("instansi_perkiraan", ""),
                    entry.get("kecamatan_perkiraan", ""),
                ),
            )
            conn.commit()
        finally:
            conn.close()


def load_complaints():
    """Return list[dict], urut id."""
    init_db()
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM complaints ORDER BY id").fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]
