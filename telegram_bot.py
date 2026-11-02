import json
import os
import random
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

import db
from ask_rag import (
    answer_query,
    detect_query_kecamatan,
    extract_complaint_focus_terms,
    format_regulation_candidate_label,
    get_structured_dataset,
    infer_disposition_candidates,
    normalize_text,
    retrieve_context,
)

load_dotenv()


BASE_DIR = Path(__file__).resolve().parent
STATE_PATH = BASE_DIR / "telegram_bot_state.json"
COMPLAINTS_LOG_PATH = BASE_DIR / "complaints_log.jsonl"
PHOTOS_DIR = Path(os.getenv("PHOTOS_DIR", str(BASE_DIR / "complaint_photos")))
TELEGRAM_API_BASE = "https://api.telegram.org"
DEFAULT_MODEL = os.getenv("TELEGRAM_MODEL_NAME", "qwen2.5:7b")
DEFAULT_TOP_K = int(os.getenv("TELEGRAM_TOP_K", "5"))
DEFAULT_TEMPERATURE = float(os.getenv("TELEGRAM_TEMPERATURE", "0.1"))
LONG_POLL_TIMEOUT = int(os.getenv("TELEGRAM_LONG_POLL_TIMEOUT", "30"))
MAX_HISTORY_MESSAGES = int(os.getenv("TELEGRAM_MAX_HISTORY", "12"))
MAX_MESSAGE_LENGTH = 3500
ADMIN_GROUP_CHAT_ID = os.getenv("TELEGRAM_ADMIN_GROUP_ID", "-5455643376").strip()


def load_state():
    """Baca state bot dari telegram_bot.db (SQLite). Lihat db.py untuk detail."""
    return db.load_state()


def save_state(state):
    """Simpan state bot ke telegram_bot.db (SQLite, atomik via transaksi)."""
    db.save_state(state)


def get_bot_token():
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "Environment variable TELEGRAM_BOT_TOKEN belum diisi."
        )
    return token


def telegram_request(method, payload=None, timeout=60):
    token = get_bot_token()
    url = f"{TELEGRAM_API_BASE}/bot{token}/{method}"
    response = requests.post(url, json=payload or {}, timeout=timeout)
    response.raise_for_status()

    data = response.json()
    if not data.get("ok"):
        raise RuntimeError(
            f"Telegram API error pada {method}: {data}"
        )
    return data["result"]


def download_telegram_photo(file_id, dest_stem):
    """Download foto dari Telegram (via file_id) ke PHOTOS_DIR. Return Path lokal, atau None kalau gagal."""
    try:
        file_info = telegram_request("getFile", payload={"file_id": file_id}, timeout=30)
        remote_path = file_info["file_path"]
        suffix = Path(remote_path).suffix or ".jpg"
        PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
        dest_path = PHOTOS_DIR / f"{dest_stem}{suffix}"
        token = get_bot_token()
        url = f"{TELEGRAM_API_BASE}/file/bot{token}/{remote_path}"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        dest_path.write_bytes(response.content)
        return dest_path
    except Exception as exc:
        print(f"[WARN] Gagal download foto aduan ({file_id}): {exc}")
        return None


def get_updates(offset=None, timeout=LONG_POLL_TIMEOUT):
    payload = {
        "timeout": timeout,
        "allowed_updates": ["message"],
    }
    if offset is not None:
        payload["offset"] = offset

    return telegram_request(
        "getUpdates",
        payload=payload,
        timeout=timeout + 10,
    )


def send_chat_action(chat_id, action="typing"):
    return telegram_request(
        "sendChatAction",
        payload={
            "chat_id": chat_id,
            "action": action,
        },
        timeout=30,
    )


def split_message(text, limit=MAX_MESSAGE_LENGTH):
    compact = str(text).strip()
    if len(compact) <= limit:
        return [compact]

    parts = []
    remaining = compact
    while len(remaining) > limit:
        split_at = remaining.rfind("\n\n", 0, limit)
        if split_at > 0:
            parts.append(remaining[:split_at].strip())
            remaining = remaining[split_at:].strip()
            continue

        split_at = remaining.rfind("\n", 0, limit)
        if split_at <= 0:
            split_at = remaining.rfind(" ", 0, limit)
        if split_at <= 0:
            split_at = limit
        parts.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()

    if remaining:
        parts.append(remaining)
    return parts


def send_message(chat_id, text, reply_to_message_id=None):
    parts = split_message(text)
    last_result = None
    for index, part in enumerate(parts):
        payload = {
            "chat_id": chat_id,
            "text": part,
        }
        if reply_to_message_id and index == 0:
            payload["reply_to_message_id"] = reply_to_message_id
        last_result = telegram_request(
            "sendMessage",
            payload=payload,
            timeout=60,
        )
    return last_result


def get_chat_history(state, chat_id):
    histories = state.setdefault("chat_histories", {})
    return histories.setdefault(str(chat_id), [])


def append_chat_message(
    state,
    chat_id,
    role,
    content,
    sources=None,
    mode=None,
    disposition_candidates=None,
    resolved_jenis=None,
    route_mode=None,
):
    history = get_chat_history(state, chat_id)
    history.append(
        {
            "role": role,
            "content": str(content).strip(),
            "sources": sources or [],
            "mode": mode,
            "disposition_candidates": disposition_candidates or [],
            "resolved_jenis": resolved_jenis,
            "route_mode": route_mode,
        }
    )
    if len(history) > MAX_HISTORY_MESSAGES:
        del history[:-MAX_HISTORY_MESSAGES]


def reset_chat_history(state, chat_id):
    state.setdefault("chat_histories", {})[str(chat_id)] = []


WIB = timezone(timedelta(hours=7))


def _wib_now():
    """Return current time as ISO string in WIB (UTC+7)."""
    ts = time.time()
    now = datetime.fromtimestamp(ts, tz=WIB)
    return now.strftime("%Y-%m-%dT%H:%M:%S+07:00"), ts


def infer_kategori_perkiraan(masalah_text):
    try:
        focus_terms = extract_complaint_focus_terms(masalah_text)
        if not focus_terms:
            return ""
        df = get_structured_dataset()
        base_mask = df["kategori"].str.strip() != ""

        def combine_term_masks(combine_op):
            mask = None
            for term in focus_terms:
                term_mask = df["_search_text"].str.contains(re.escape(term), regex=True)
                mask = term_mask if mask is None else combine_op(mask, term_mask)
            return mask

        filtered = df[base_mask & combine_term_masks(lambda a, b: a & b)]
        if filtered.empty:
            filtered = df[base_mask & combine_term_masks(lambda a, b: a | b)]

        if filtered.empty:
            return ""
        return str(filtered["kategori"].value_counts().idxmax()).strip()
    except Exception as exc:
        print(f"[WARN] Gagal infer kategori_perkiraan: {exc}")
        return ""


def infer_instansi_perkiraan(masalah_text):
    try:
        candidates = infer_disposition_candidates(masalah_text, limit=1)
        return candidates[0]["instansi"] if candidates else ""
    except Exception as exc:
        print(f"[WARN] Gagal infer instansi_perkiraan: {exc}")
        return ""


def infer_kecamatan_perkiraan(lokasi_text):
    try:
        return detect_query_kecamatan(lokasi_text) or ""
    except Exception as exc:
        print(f"[WARN] Gagal infer kecamatan_perkiraan: {exc}")
        return ""


def log_complaint(complaint_data: dict, chat_id, session_code: str, foto_path=None):
    """Simpan satu aduan ke tabel `complaints` di telegram_bot.db (lihat db.py).

    kategori/instansi/kecamatan diisi otomatis via kemiripan ke data historis
    (lihat infer_*_perkiraan di atas) -- perkiraan, bukan klasifikasi resmi.
    """
    iso_str, unix_ts = _wib_now()
    masalah = complaint_data.get("masalah", "")
    lokasi = complaint_data.get("lokasi", "")
    entry = {
        "kode": session_code,
        "timestamp": iso_str,
        "timestamp_unix": unix_ts,
        "chat_id": str(chat_id),
        "masalah": masalah,
        "lokasi": lokasi,
        "waktu_kejadian": complaint_data.get("waktu", ""),
        "has_foto": complaint_data.get("has_foto", False),
        "foto_path": str(foto_path) if foto_path else "",
        "tahun": int(iso_str[:4]),
        "kategori_perkiraan": infer_kategori_perkiraan(masalah),
        "instansi_perkiraan": infer_instansi_perkiraan(masalah),
        "kecamatan_perkiraan": infer_kecamatan_perkiraan(lokasi),
    }
    try:
        db.log_complaint_row(entry)
    except Exception as exc:
        print(f"[WARN] Gagal tulis aduan ke telegram_bot.db: {exc}")
    return entry


def notify_admin_new_complaint(complaint_data: dict, chat_id, session_code: str, iso_str: str):
    """Kirim notifikasi aduan baru ke grup admin operator Diskominfo."""
    if not ADMIN_GROUP_CHAT_ID:
        return
    has_foto = complaint_data.get("has_foto", False)
    foto_note = " 📷" if has_foto else ""
    msg = (
        f"🔔 *Aduan Baru Masuk*\n\n"
        f"📋 Masalah : {complaint_data.get('masalah', '-')}{foto_note}\n"
        f"📍 Lokasi  : {complaint_data.get('lokasi', '-')}\n"
        f"📅 Waktu   : {complaint_data.get('waktu', '-')}\n\n"
        f"🕐 Masuk   : {iso_str.replace('T', ' ').replace('+07:00', ' WIB')}\n"
        f"🔑 Kode    : `{session_code}`\n"
        f"💬 Chat ID : `{chat_id}`"
    )
    try:
        telegram_request(
            "sendMessage",
            payload={
                "chat_id": ADMIN_GROUP_CHAT_ID,
                "text": msg,
                "parse_mode": "Markdown",
            },
        )
    except Exception as exc:
        print(f"[WARN] Gagal notif admin: {exc}")


CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
CODE_LENGTH = 6
CANDIDATE_LIMIT = 5
CONFIRM_REJECT_WORDS = {"bukan", "tidak", "tidak ada", "none", "gak ada", "ga ada", "bukan semua"}


def get_session_codes(state):
    return state.setdefault("session_codes", {})


def get_active_code_map(state):
    return state.setdefault("active_code_by_chat", {})


def get_pending_confirmations(state):
    return state.setdefault("pending_historical_confirmation", {})


def generate_session_code(existing_codes):
    while True:
        code = "".join(random.choices(CODE_ALPHABET, k=CODE_LENGTH))
        if code not in existing_codes:
            return code


def summarize_context_for_code(question, answer):
    return {
        "question": " ".join(str(question).split())[:300],
        "answer": " ".join(str(answer).split())[:600],
    }


def create_session_code_for_chat(state, chat_id, question, answer, linked_source_id=None, complaint_data=None):
    codes = get_session_codes(state)
    code = generate_session_code(set(codes.keys()))
    entry = {
        "chat_id": str(chat_id),
        "created_at": time.time(),
        "context": summarize_context_for_code(question, answer),
        "linked_source_id": linked_source_id,
    }
    if complaint_data:
        entry["complaint_data"] = {
            "masalah": complaint_data.get("masalah", ""),
            "lokasi": complaint_data.get("lokasi", ""),
            "waktu_kejadian": complaint_data.get("waktu", ""),
            "has_foto": complaint_data.get("has_foto", False),
        }
    codes[code] = entry
    get_active_code_map(state)[str(chat_id)] = code
    return code


RESUME_CODE_PATTERN = re.compile(
    r"\b(?:kode|lanjut(?:kan)?|resume|sambung(?:kan)?)\b[^A-Za-z0-9]{0,10}([A-Z0-9]{6})\b",
    re.IGNORECASE,
)


def detect_resume_code_request(text):
    match = RESUME_CODE_PATTERN.search(str(text or ""))
    if match:
        return match.group(1).upper()
    return None


def resolve_resume_code(state, chat_id, code):
    entry = get_session_codes(state).get(code)
    if not entry:
        return None
    if str(entry.get("chat_id")) != str(chat_id):
        return None
    return entry


HISTORICAL_RECONNECT_MARKERS = {
    "dulu saya", "sebelumnya saya", "pernah ngadu", "pernah lapor", "pernah mengadu",
    "aduan saya sebelumnya", "laporan saya dulu", "aduan lama saya", "aduan saya yang lama",
    "pengaduan saya dulu", "pengaduan saya sebelumnya",
}


def detect_historical_reconnect_intent(text):
    text_norm = normalize_text(text)
    return any(marker in text_norm for marker in HISTORICAL_RECONNECT_MARKERS)


def search_historical_candidates(query, limit=CANDIDATE_LIMIT):
    items = retrieve_context(
        query=query,
        top_k=max(limit * 2, 10),
        jenis="pengaduan",
        tipe_dokumen=None,
        source_id=None,
    )
    seen_source_ids = set()
    candidates = []
    for item in items:
        metadata = item.get("metadata") or {}
        source_id = str(metadata.get("source_id", "")).strip()
        if not source_id or source_id in seen_source_ids:
            continue
        seen_source_ids.add(source_id)
        candidates.append(
            {
                "source_id": source_id,
                "judul": compact_public_label(metadata.get("judul", "-")),
                "kategori": compact_public_label(metadata.get("kategori", "-")),
                "kecamatan": compact_public_label(metadata.get("kecamatan", "-")),
                "tahun": metadata.get("tahun", "-"),
            }
        )
        if len(candidates) >= limit:
            break
    return candidates


def format_historical_candidates_message(candidates):
    lines = ["Saya menemukan beberapa aduan lama yang mungkin cocok:", ""]
    for index, candidate in enumerate(candidates, start=1):
        lines.append(
            f"{index}. {candidate['judul']} -- {candidate['kategori']}, "
            f"Kec. {candidate['kecamatan']}, {candidate['tahun']}"
        )
    lines.append("")
    lines.append("Balas dengan nomornya kalau ada yang sesuai, atau ketik \"bukan\" kalau tidak ada.")
    return "\n".join(lines)


def parse_candidate_selection(text, n_candidates):
    stripped = str(text or "").strip().lower()
    if any(word in stripped for word in CONFIRM_REJECT_WORDS):
        return "reject"
    match = re.match(r"^(\d+)\b", stripped)
    if match:
        index = int(match.group(1))
        if 1 <= index <= n_candidates:
            return index
    return None


def extract_message_text(message):
    return (
        message.get("text")
        or message.get("caption")
        or ""
    ).strip()


def format_disposition_candidates(candidates):
    if not candidates:
        return ""

    lines = ["", "Kandidat instansi:"]
    for index, candidate in enumerate(candidates, start=1):
        lines.append(f"{index}. {candidate['instansi']}")
    return "\n".join(lines)


def compact_public_label(value, limit=180):
    compact = " ".join(str(value or "").split()).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def build_public_source_label(item):
    metadata = item.get("metadata") or {}
    jenis = str(metadata.get("jenis", "")).strip().lower()
    tipe_dokumen = str(metadata.get("tipe_dokumen", "")).strip().lower()
    dokumen = compact_public_label(metadata.get("dokumen", ""))
    section = compact_public_label(metadata.get("section", ""))

    if jenis == "pengaduan":
        kategori = compact_public_label(metadata.get("kategori", ""))
        instansi = compact_public_label(metadata.get("instansi", ""))
        details = []
        if kategori:
            details.append(f"kategori: {kategori}")
        if instansi:
            details.append(f"instansi: {instansi}")
        suffix = f" ({'; '.join(details)})" if details else ""
        return f"Data historis SP4N-LAPOR! Kabupaten Bandung{suffix}"

    if tipe_dokumen == "sop":
        label = dokumen or "SOP Penyelesaian Pengaduan SP4N-LAPOR"
    else:
        label = compact_public_label(format_regulation_candidate_label(item))

    if section and section not in {"-", "Dokumen Lengkap"}:
        label += f" - {section}"
    return label


def collect_public_source_labels(context_items, limit=None):
    labels = []
    seen = set()

    for item in context_items or []:
        label = build_public_source_label(item)
        if not label or label in seen:
            continue
        seen.add(label)
        labels.append(label)
        if limit is not None and len(labels) >= limit:
            break

    return labels


def format_public_sources(context_items, limit=3):
    labels = collect_public_source_labels(context_items, limit=None)

    if not labels:
        return ""

    visible_labels = labels[:limit]
    lines = ["", "Sumber ringkas:"]
    lines.extend(f"- {label}" for label in visible_labels)
    if len(labels) > limit:
        lines.append(f"- ... dan {len(labels) - limit} sumber lain.")
    return "\n".join(lines)


def strip_embedded_source_note(answer):
    stripped = str(answer or "").strip()
    return re.sub(
        r"\n\s*\n?(?:Sumber|Rujukan)\s*:\s*.+$",
        "",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()


def strip_internal_ids(answer):
    """Buang identitas internal (chunk_id) yang kadang ikut tercetak model.

    Hanya kosmetik antarmuka: dijalankan saat memformat jawaban untuk Telegram,
    tidak menyentuh jalur generation/evaluasi. Identitas sumber tetap tampil pada
    baris "Sumber" yang disusun terpisah.
    """
    text = str(answer or "")
    text = re.sub(
        r"\baduan dengan chunk_id:\s*[\w-]+",
        "aduan tersebut",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\(?\s*chunk_id:\s*[\w-]+\s*\)?", "", text, flags=re.IGNORECASE)

    text = text.strip()
    original = text
    preamble_patterns = [
        r"^(?:Pengaduan|Aduan) utama[^.\n]*\.\s*",
        r"^Judul aduan[^.\n]*\.\s*",
        r"^[^.\n]*jawaban yang dapat disampaikan adalah\s*:?\s*",
        r"^Berdasarkan aduan tersebut,\s*",
    ]
    for _ in range(3):
        before = text
        for pat in preamble_patterns:
            text = re.sub(pat, "", text, flags=re.IGNORECASE)
        text = text.strip()
        if text == before:
            break
    if len(text) < 20:
        text = original
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def format_structured_sources(result, source_labels):
    if not source_labels:
        return ""

    jenis = result.get("resolved_jenis")

    if jenis == "regulasi":
        shown = source_labels[:2]
        return "\n\nSumber: " + "; ".join(shown)

    return "\n\nSumber: " + source_labels[0]


def format_result_text(result):
    raw_answer = result.get("answer", "")
    answer = strip_internal_ids(strip_embedded_source_note(raw_answer))
    from ask_rag import _CHANNEL_INFO_SUFFIX
    if _CHANNEL_INFO_SUFFIX.strip() in raw_answer and _CHANNEL_INFO_SUFFIX.strip() not in answer:
        answer = answer.rstrip() + _CHANNEL_INFO_SUFFIX
    disposition_candidates = result.get("disposition_candidates") or []
    if result.get("mode") == "disposition" and disposition_candidates:
        answer += format_disposition_candidates(disposition_candidates)

    modes_without_public_sources = {
        "acknowledgement",
        "beneficiary_distribution_unsupported",
        "chat",
        "chat_fallback",
        "complaint_flow_collecting",
        "complaint_flow_done",
        "complaint_flow_start",
        "disposition",
        "identity_lookup_declined",
        "insufficient_context",
        "need_clarification",
        "new_complaint_reset",
        "out_of_scope_region",
        "out_of_scope_service",
        "pasal_not_found",
        "regulation_clarify",
        "service_info",
        "service_info_handoff",
        "tracking_not_found",
        "operator_handoff",
        "out_of_scope_topic",
        "jurisdiction_hardstop",
    }
    source_items = result.get("context_items") or []
    public_source_limit = 5 if result.get("resolved_jenis") == "regulasi" else 3
    source_labels = collect_public_source_labels(source_items, limit=public_source_limit)

    if (
        result.get("mode") not in modes_without_public_sources
    ):
        source_block = format_structured_sources(result, source_labels)
        if source_block:
            return f"{answer}\n{source_block}".strip()

    return answer.strip()


def build_help_text():
    return (
        "Bot pengaduan aktif.\n\n"
        "Perintah:\n"
        "/start - info singkat\n"
        "/help - bantuan\n"
        "/reset - hapus riwayat chat bot\n\n"
        "Contoh:\n"
        "- Bagaimana alur disposisi pengaduan di SP4N-LAPOR?\n"
        "- Masalah Jalan berlubang di Soreang harus komplain ke dinas mana?\n"
        "- Apa dasar hukum pengelolaan pengaduan?\n"
    )


def handle_command(text, state, chat_id):
    command = text.split()[0].lower()

    if command in {"/start", "/help"}:
        return build_help_text()

    if command == "/reset":
        reset_chat_history(state, chat_id)
        return "Riwayat percakapan untuk chat ini sudah dihapus."

    return (
        "Perintah tidak dikenali. Gunakan /help untuk melihat panduan."
    )


def get_complaint_flow_states(state):
    return state.setdefault("complaint_flow_states", {})


def process_message(message, state):
    chat = message.get("chat") or {}
    chat_id = chat.get("id")
    if not chat_id:
        return

    has_photo = bool(message.get("photo") or message.get("document"))
    text = extract_message_text(message)

    def _get_file_id(msg):
        photos = msg.get("photo")
        if photos:
            return photos[-1]["file_id"]
        doc = msg.get("document")
        return doc.get("file_id") if doc else None

    if has_photo:
        file_id = _get_file_id(message)
        flow = get_complaint_flow_states(state).get(str(chat_id))

        if flow:
            flow["has_foto"] = True
            flow["foto_file_id"] = file_id
            get_complaint_flow_states(state)[str(chat_id)] = flow
            save_state(state)

            step = flow.get("step", "")
            if step in {"lokasi", "waktu"}:
                caption_note = f'\nCaption: "{text}"' if text else ""
                next_ask = (
                    "Sekarang, boleh sebutkan lokasi kejadiannya? "
                    "(contoh: Jl. Merdeka, Kec. Cileunyi)"
                    if step == "lokasi"
                    else "Sejak kapan masalah ini terjadi?"
                )
                send_message(
                    chat_id,
                    f"Bukti foto diterima ✓{caption_note}\n{next_ask}",
                    reply_to_message_id=message.get("message_id"),
                )
                return
            else:
                send_message(
                    chat_id,
                    "Bukti foto diterima ✓ Akan dicatat dalam ringkasan laporan.",
                    reply_to_message_id=message.get("message_id"),
                )
                if not text:
                    return
        else:
            if not text:
                send_message(
                    chat_id,
                    "Foto diterima. Ketik deskripsi masalahnya terlebih dahulu, "
                    "foto bisa disertakan saat melapor ke SP4N-LAPOR! resmi.",
                    reply_to_message_id=message.get("message_id"),
                )
                return

    if not text:
        send_message(
            chat_id,
            "Saat ini bot hanya memproses pesan teks atau foto.",
            reply_to_message_id=message.get("message_id"),
        )
        return

    if text.startswith("/"):
        if text.strip().lower() == "/reset":
            get_complaint_flow_states(state).pop(str(chat_id), None)
        reply_text = handle_command(text, state, chat_id)
        send_message(
            chat_id,
            reply_text,
            reply_to_message_id=message.get("message_id"),
        )
        save_state(state)
        return

    append_chat_message(state, chat_id, "user", text)
    save_state(state)

    pending = get_pending_confirmations(state).get(str(chat_id))
    if pending:
        selection = parse_candidate_selection(text, len(pending["candidates"]))
        get_pending_confirmations(state).pop(str(chat_id), None)

        if selection == "reject":
            reply_text = "Baik, tidak masalah. Silakan ceritakan detail aduan Anda, nanti saya bantu dari awal."
            append_chat_message(state, chat_id, "assistant", reply_text)
            save_state(state)
            send_message(chat_id, reply_text, reply_to_message_id=message.get("message_id"))
            return

        if isinstance(selection, int):
            chosen = pending["candidates"][selection - 1]
            code = create_session_code_for_chat(
                state, chat_id,
                question=f"Konfirmasi aduan lama: {chosen['judul']}",
                answer=f"Kategori: {chosen['kategori']}, Kecamatan: {chosen['kecamatan']}, Tahun: {chosen['tahun']}",
                linked_source_id=chosen["source_id"],
            )
            reply_text = (
                f"Baik, aduan '{chosen['judul']}' sudah tersambung.\n\n"
                f"Catat kode ini: {code}\n"
                "Sebutkan kode ini kapan pun untuk melanjutkan konteks aduan ini, "
                "walau beberapa hari/minggu ke depan."
            )
            append_chat_message(state, chat_id, "assistant", reply_text)
            save_state(state)
            send_message(chat_id, reply_text, reply_to_message_id=message.get("message_id"))
            return

    resumed_context = None
    resume_code = detect_resume_code_request(text)
    if resume_code:
        entry = resolve_resume_code(state, chat_id, resume_code)
        if entry is None:
            reply_text = (
                "Kode tidak ditemukan atau tidak valid untuk chat ini. "
                "Pastikan kode benar, atau mulai obrolan baru."
            )
            append_chat_message(state, chat_id, "assistant", reply_text)
            save_state(state)
            send_message(chat_id, reply_text, reply_to_message_id=message.get("message_id"))
            return
        resumed_context = entry["context"]

    if resumed_context is None and detect_historical_reconnect_intent(text):
        candidates = search_historical_candidates(text)
        if candidates:
            get_pending_confirmations(state)[str(chat_id)] = {
                "candidates": candidates,
                "created_at": time.time(),
            }
            reply_text = format_historical_candidates_message(candidates)
            append_chat_message(state, chat_id, "assistant", reply_text)
            save_state(state)
            send_message(chat_id, reply_text, reply_to_message_id=message.get("message_id"))
            return

    send_chat_action(chat_id, action="typing")

    try:
        chat_history = get_chat_history(state, chat_id)
        if resumed_context:
            chat_history = [
                {"role": "user", "content": resumed_context["question"]},
                {"role": "assistant", "content": resumed_context["answer"]},
            ] + chat_history

        active_flow = get_complaint_flow_states(state).get(str(chat_id))

        result = answer_query(
            query=text,
            model_name=DEFAULT_MODEL,
            top_k=DEFAULT_TOP_K,
            temperature=DEFAULT_TEMPERATURE,
            chat_history=chat_history,
            complaint_flow_state=active_flow,
        )
        reply_text = format_result_text(result)

        new_flow = result.get("complaint_flow_state")
        if new_flow:
            get_complaint_flow_states(state)[str(chat_id)] = new_flow
        elif result.get("mode") == "complaint_flow_done":
            get_complaint_flow_states(state).pop(str(chat_id), None)

            complaint_data = result.get("complaint_data") or {}
            if complaint_data.get("masalah"):
                code = create_session_code_for_chat(
                    state, chat_id,
                    question=complaint_data.get("masalah", ""),
                    answer=reply_text,
                    complaint_data=complaint_data,
                )
                foto_file_id = complaint_data.get("foto_file_id")
                local_foto_path = (
                    download_telegram_photo(foto_file_id, code) if foto_file_id else None
                )
                log_entry = log_complaint(complaint_data, chat_id, code, foto_path=local_foto_path)
                notify_admin_new_complaint(
                    complaint_data, chat_id, code,
                    log_entry["timestamp"],
                )
                if foto_file_id and ADMIN_GROUP_CHAT_ID:
                    try:
                        telegram_request("sendPhoto", payload={
                            "chat_id": ADMIN_GROUP_CHAT_ID,
                            "photo": foto_file_id,
                            "caption": f"📷 Bukti foto aduan {code}\n{complaint_data.get('masalah', '')}",
                        })
                    except Exception as exc:
                        print(f"[WARN] Gagal forward foto ke admin: {exc}")
                reply_text += (
                    f"\n\n🔑 Kode referensi aduan Anda: *{code}*\n"
                    "Simpan kode ini — bisa dipakai untuk melanjutkan "
                    "atau menanyakan status aduan ini nanti."
                )
        elif result.get("mode") == "complaint_flow_start":
            get_complaint_flow_states(state)[str(chat_id)] = result.get("complaint_flow_state", {})

        if result.get("mode") in ("operator_handoff", "rag_low_confidence") and ADMIN_GROUP_CHAT_ID:
            try:
                from datetime import datetime, timezone, timedelta
                _now = datetime.now(timezone(timedelta(hours=7)))
                _ts = _now.strftime("%Y-%m-%d %H:%M:%S WIB")
                _history = get_chat_history(state, chat_id)
                _context_lines = []
                for msg in _history[-6:]:
                    _role = "User" if msg["role"] == "user" else "Bot"
                    _content = str(msg.get("content", ""))[:200]
                    _context_lines.append(f"  {_role}: {_content}")
                _context_summary = "\n".join(_context_lines) if _context_lines else "(tidak ada riwayat)"
                _bot_answer = str(result.get("answer", ""))[:300]
                _op_msg = (
                    "🔔 *Eskalasi — Bot Tidak Dapat Menjawab*\n\n"
                    f"💬 Pertanyaan: {text}\n"
                    f"🤖 Respons bot: {_bot_answer}\n"
                    f"🕐 Waktu: {_ts}\n"
                    f"👤 Chat ID: `{chat_id}`\n\n"
                    f"📋 *Riwayat percakapan:*\n{_context_summary}"
                )
                telegram_request("sendMessage", payload={
                    "chat_id": ADMIN_GROUP_CHAT_ID,
                    "text": _op_msg,
                    "parse_mode": "Markdown",
                })
            except Exception as exc:
                print(f"[WARN] Gagal notif admin operator handoff: {exc}")

        if result.get("resolved_jenis") == "pengaduan" and str(chat_id) not in get_active_code_map(state):
            code = create_session_code_for_chat(state, chat_id, question=text, answer=result.get("answer", ""))
            reply_text += (
                f"\n\nCatat kode ini: {code}\n"
                "Sebutkan kode ini kapan pun untuk melanjutkan obrolan ini nanti."
            )

        append_chat_message(
            state,
            chat_id,
            "assistant",
            result.get("answer", ""),
            sources=result.get("context_items", []),
            mode=result.get("mode"),
            disposition_candidates=result.get("disposition_candidates", []),
            resolved_jenis=result.get("resolved_jenis"),
            route_mode=result.get("route_mode"),
        )
        save_state(state)
    except Exception as exc:
        reply_text = f"Gagal menjalankan agen pengaduan: {exc}"

    send_message(
        chat_id,
        reply_text,
        reply_to_message_id=message.get("message_id"),
    )


def run_polling():
    state = load_state()
    offset = state.get("last_update_id")

    print("[START] Telegram bot polling aktif")
    print(f"[INFO] State database: {db.DB_PATH}")
    print(f"[INFO] Model: {DEFAULT_MODEL}")
    print(f"[INFO] Top K: {DEFAULT_TOP_K}")
    print(f"[INFO] Poll timeout: {LONG_POLL_TIMEOUT}s")

    while True:
        try:
            updates = get_updates(offset=offset)
            for update in updates:
                update_id = update.get("update_id")
                if update_id is not None:
                    offset = update_id + 1
                    state["last_update_id"] = offset

                message = update.get("message")
                if not message:
                    continue

                process_message(message, state)
                save_state(state)
        except KeyboardInterrupt:
            print("[STOP] Telegram bot dihentikan.")
            break
        except Exception as exc:
            print(f"[ERROR] {exc}")
            time.sleep(5)


if __name__ == "__main__":
    run_polling()