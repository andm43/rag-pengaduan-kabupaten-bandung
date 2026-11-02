
from datetime import datetime, timedelta, timezone
import hashlib
import html
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import db
import pipeline_manager
import re

from search_corpus import (
    build_options,
    decode_markdown,
    find_body_start,
    find_corpus_dirs,
    read_frontmatter,
    scan_text,
)


BASE_DIR = Path(__file__).resolve().parent
VERIFICATION_LOG_PATH = BASE_DIR / "verifikasi_jawaban.json"

SHOW_VERIFIKASI_TAB = False

WIB = timezone(timedelta(hours=7))
ACCENT_LIGHT = "#2a78d6"
ACCENT_DARK = "#3987e5"
TOP_KATEGORI = 10

ICON_CDN = "https://cdn.jsdelivr.net/npm/@tabler/icons@3.46.0/icons/outline"


def icon_img(name, size="1.05em", opacity=0.85):
    """<img> ke SVG icon dari CDN. Dipakai di semua tempat yang menerima HTML
    (st.markdown dengan unsafe_allow_html=True) -- st.tabs/st.metric/st.radio
    Streamlit tidak merender HTML di labelnya, jadi tidak dipakai di situ.
    filter: invert(1) -- SVG asli bergaris hitam, di-invert jadi putih supaya
    kontras di background gelap (bukan hitam pudar di atas panel gelap)."""
    return (
        f'<img src="{ICON_CDN}/{name}.svg" alt="" '
        f'style="width:{size};height:{size};vertical-align:-0.2em;'
        f'margin-right:0.35rem;opacity:{opacity};filter:invert(1)" />'
    )


def section_heading(name, text):
    """Pengganti st.subheader() supaya bisa menyisipkan icon CDN (st.subheader
    tidak mendukung HTML)."""
    st.markdown(
        f'<div class="section-heading">{icon_img(name, size="1.2em")}{text}</div>',
        unsafe_allow_html=True,
    )


def metric_card(name, label, value, help_text=None):
    """Pengganti st.metric() supaya bisa menyisipkan icon CDN (st.metric tidak
    mendukung HTML di labelnya)."""
    help_html = f'<div class="metric-help">{help_text}</div>' if help_text else ""
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">{icon_img(name, size="0.95em")}{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'{help_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="Dashboard Admin Aduan Telegram",
    page_icon=f"{ICON_CDN}/layout-dashboard.svg",
    layout="wide",
)

DASHBOARD_CSS = """
<style>
/* Sidebar: panel gelap solid (bukan overlay putih 2-3% yang nyaris tak
kelihatan di atas hitam) -- kontrasnya jelas terhadap background utama,
dan ruang kosong di bawah konten ditutup footer lewat flex margin-top:auto. */
[data-testid="stSidebar"] {
    background-color: #1E293B;
    border-right: 1px solid #334155;
}
[data-testid="stSidebar"] > div:first-child {
    display: flex;
    flex-direction: column;
    min-height: 100vh;
}
[data-testid="stSidebar"] .block-container {
    padding-top: 1.25rem;
    flex: 1 0 auto;
}
.sidebar-brand {
    padding: 0 0 1.1rem 0;
    border-bottom: 1px solid #334155;
    margin-bottom: 1rem;
}
.sidebar-brand-title {
    font-size: 1.1rem;
    font-weight: 700;
    margin: 0;
    line-height: 1.3;
    color: #F1F5F9;
}
.sidebar-brand-sub {
    font-size: 0.8rem;
    color: #94A3B8;
    margin-top: 0.2rem;
}
.sidebar-section-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #94A3B8;
    margin: 1.15rem 0 0.5rem 0;
}
.sidebar-footer {
    margin-top: auto;
    padding: 1rem 0 0.5rem 0;
    border-top: 1px solid #334155;
    font-size: 0.72rem;
    color: #64748B;
}
div[data-testid="stMetric"] {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 0.7rem 0.9rem;
}
h1 { letter-spacing: -0.01em; color: #F1F5F9; }
.app-tagline {
    color: #94A3B8;
    font-size: 0.92rem;
    margin-top: -0.6rem;
    margin-bottom: 1rem;
}
div[data-testid="stTabs"] button[role="tab"] {
    font-weight: 600;
}
.section-heading {
    display: flex;
    align-items: center;
    font-size: 1.5rem;
    font-weight: 600;
    line-height: 1.25;
    margin: 0.3rem 0 0.7rem 0;
    color: #F1F5F9;
}
.metric-card {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 0.7rem 0.9rem;
    height: 100%;
}
.metric-card .metric-label {
    display: flex;
    align-items: center;
    font-size: 0.8rem;
    color: #94A3B8;
    margin-bottom: 0.2rem;
}
.metric-card .metric-value {
    font-size: 1.7rem;
    font-weight: 600;
    line-height: 1.2;
    color: #F1F5F9;
}
.metric-card .metric-help {
    font-size: 0.72rem;
    color: #64748B;
    margin-top: 0.15rem;
}
/* Icon CDN di tab -- st.tabs() tidak menerima HTML di labelnya, jadi icon
disisipkan lewat ::before + background-image, dikunci per posisi tab. */
div[data-testid="stTabs"] div[role="tablist"] button[role="tab"]:nth-of-type(1)::before,
div[data-testid="stTabs"] div[role="tablist"] button[role="tab"]:nth-of-type(2)::before,
div[data-testid="stTabs"] div[role="tablist"] button[role="tab"]:nth-of-type(3)::before,
div[data-testid="stTabs"] div[role="tablist"] button[role="tab"]:nth-of-type(4)::before {
    content: "";
    display: inline-block;
    width: 1.05em;
    height: 1.05em;
    margin-right: 0.4rem;
    vertical-align: -0.2em;
    background-repeat: no-repeat;
    background-position: center;
    background-size: contain;
    opacity: 0.85;
    filter: invert(1);
}
div[data-testid="stTabs"] div[role="tablist"] button[role="tab"]:nth-of-type(1)::before {
    background-image: url('""" + ICON_CDN + """/clipboard-list.svg');
}
div[data-testid="stTabs"] div[role="tablist"] button[role="tab"]:nth-of-type(2)::before {
    background-image: url('""" + ICON_CDN + """/database.svg');
}
div[data-testid="stTabs"] div[role="tablist"] button[role="tab"]:nth-of-type(3)::before {
    background-image: url('""" + ICON_CDN + """/shield-check.svg');
}
div[data-testid="stTabs"] div[role="tablist"] button[role="tab"]:nth-of-type(4)::before {
    background-image: url('""" + ICON_CDN + """/settings-automation.svg');
}

/* ---- Polish menyeluruh: input, tombol, container, badge ---- */

/* Highlight hasil pencarian -- default browser <mark> kuning terang
tabrakan keras dengan tema gelap; ganti jadi aksen biru transparan. */
mark {
    background-color: rgba(57, 135, 229, 0.32);
    color: #F1F5F9;
    padding: 0 0.2em;
    border-radius: 3px;
}

/* Text input, text area, dan select (BaseWeb) -- disamakan warnanya dengan
kartu (bg lebih gelap dari page, border slate) supaya tidak terasa seperti
komponen default yang nempel asal di atas tema gelap. */
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea,
div[data-testid="stNumberInput"] input,
div[data-baseweb="select"] > div,
div[data-baseweb="base-input"] {
    background-color: #0F172A !important;
    border-color: #334155 !important;
    border-radius: 8px !important;
}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stTextArea"] textarea:focus {
    border-color: #3987e5 !important;
    box-shadow: 0 0 0 1px #3987e5 !important;
}

/* Tag multiselect (mis. daftar Basis Pengetahuan) -- default BaseWeb pakai
merah muda, diganti aksen biru supaya konsisten satu palet dengan sisanya. */
span[data-baseweb="tag"] {
    background-color: #3987e5 !important;
    border-radius: 6px !important;
}

/* Tombol -- radius konsisten, tombol primary pakai warna aksen. */
.stButton button, div[data-testid="stFormSubmitButton"] button {
    border-radius: 8px !important;
    font-weight: 600 !important;
}
.stButton button[kind="primary"], div[data-testid="stFormSubmitButton"] button[kind="primary"] {
    background-color: #3987e5 !important;
    border-color: #3987e5 !important;
}
.stButton button[kind="primary"]:hover, div[data-testid="stFormSubmitButton"] button[kind="primary"]:hover {
    background-color: #2f78ce !important;
    border-color: #2f78ce !important;
}

/* Semua container ber-border (st.container(border=True)) dan expander --
disamakan jadi satu bahasa "kartu": radius lebih besar, border slate,
sedikit tint di atas background supaya berlapis, bukan menyatu rata. */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 10px !important;
    border-color: #334155 !important;
}
div[data-testid="stExpander"] {
    border-radius: 8px !important;
    border-color: #334155 !important;
    overflow: hidden;
}
div[data-testid="stExpander"] summary {
    font-weight: 600 !important;
}

/* Form -- buang border/padding ganda default Streamlit (form sudah di dalam
tab yang sudah punya padding sendiri, dobel border bikin berantakan). */
div[data-testid="stForm"] {
    border: none !important;
    padding: 0 !important;
}

/* Chip/badge kecil -- dipakai untuk metadata (jenis/section) dan kata kunci
yang cocok pada mode pencarian baru, supaya tidak berupa teks polos panjang. */
.chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    margin: 0.35rem 0 0.15rem 0;
}
.chip {
    display: inline-block;
    padding: 0.16rem 0.6rem;
    border-radius: 999px;
    font-size: 0.74rem;
    background: rgba(57, 135, 229, 0.14);
    color: #7db4f5;
    border: 1px solid rgba(57, 135, 229, 0.35);
    white-space: nowrap;
}
.chip-muted {
    background: rgba(148, 163, 184, 0.1);
    color: #94A3B8;
    border-color: #334155;
}
.score-badge {
    display: inline-block;
    padding: 0.22rem 0.65rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 700;
    white-space: nowrap;
}
.score-badge-high {
    background: rgba(34, 197, 94, 0.15);
    color: #4ade80;
    border: 1px solid rgba(34, 197, 94, 0.35);
}
.score-badge-mid {
    background: rgba(234, 179, 8, 0.15);
    color: #facc15;
    border: 1px solid rgba(234, 179, 8, 0.35);
}
.score-badge-low {
    background: rgba(148, 163, 184, 0.12);
    color: #94A3B8;
    border: 1px solid #334155;
}
.hit-card-title {
    font-weight: 600;
    color: #F1F5F9;
    font-size: 0.98rem;
}
.mode-caption {
    color: #94A3B8;
    font-size: 0.86rem;
    margin: -0.35rem 0 0.9rem 0;
}
</style>
"""


def render_sidebar_brand():
    st.markdown(
        '<div class="sidebar-brand">'
        f'<p class="sidebar-brand-title">{icon_img("layout-dashboard", size="1.3em")}RAG Pengaduan</p>'
        '<p class="sidebar-brand-sub">Dashboard Admin · Kabupaten Bandung</p>'
        '</div>',
        unsafe_allow_html=True,
    )


def sidebar_section(label):
    st.markdown(f'<div class="sidebar-section-label">{label}</div>', unsafe_allow_html=True)


def accent_color():
    return ACCENT_DARK if st.get_option("theme.base") == "dark" else ACCENT_LIGHT


def file_signature(path):
    """Dipakai sebagai cache key supaya data baru (aduan/state) langsung terbaca
    setelah bot menulis ke database."""
    if not path.exists():
        return (str(path), 0, 0.0)

    stat = path.stat()
    return (str(path), stat.st_size, stat.st_mtime)


@st.cache_data(show_spinner=False)
def load_complaints(signature):
    rows = db.load_complaints()

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)

    text_columns = (
        "kode", "masalah", "lokasi", "waktu_kejadian", "chat_id", "foto_path",
        "kategori_perkiraan", "instansi_perkiraan", "kecamatan_perkiraan",
    )
    for column in text_columns:
        if column not in frame.columns:
            frame[column] = ""
        frame[column] = frame[column].fillna("").astype(str)

    if "has_foto" not in frame.columns:
        frame["has_foto"] = False
    frame["has_foto"] = frame["has_foto"].fillna(False).astype(bool)

    frame["waktu_masuk"] = pd.to_datetime(
        frame.get("timestamp"),
        format="ISO8601",
        errors="coerce",
        utc=True,
    ).dt.tz_convert(WIB)

    if frame["waktu_masuk"].isna().any() and "timestamp_unix" in frame.columns:
        fallback = pd.to_datetime(
            frame["timestamp_unix"],
            unit="s",
            errors="coerce",
            utc=True,
        ).dt.tz_convert(WIB)
        frame["waktu_masuk"] = frame["waktu_masuk"].fillna(fallback)

    frame = frame.dropna(subset=["waktu_masuk"])
    frame["tanggal"] = frame["waktu_masuk"].dt.date
    frame["jam"] = frame["waktu_masuk"].dt.hour

    return frame.sort_values("waktu_masuk", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_state(signature):
    return db.load_state()


def mask_chat_id(value):
    text = str(value)
    if len(text) <= 4:
        return "*" * len(text)
    return text[:3] + "*" * (len(text) - 5) + text[-2:]


def short(text, limit=60):
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def render_stat_tiles(frame, full_frame):
    now = datetime.now(WIB)
    last_7d = full_frame[full_frame["waktu_masuk"] >= now - timedelta(days=7)]
    with_photo = int(frame["has_foto"].sum())
    total = len(frame)
    photo_pct = f"{with_photo / total * 100:.0f}%" if total else "0%"

    tiles = [
        ("list-details", "Aduan tampil", f"{total}", "Setelah filter diterapkan"),
        ("archive", "Total sepanjang waktu", f"{len(full_frame)}", None),
        ("calendar-event", "7 hari terakhir", f"{len(last_7d)}", None),
        ("camera", "Disertai foto", photo_pct, f"{with_photo} dari {total} aduan"),
    ]
    columns = st.columns(4)
    for column, (icon_name, label, value, help_text) in zip(columns, tiles):
        with column:
            metric_card(icon_name, label, value, help_text)


def render_daily_chart(frame, color):
    if frame.empty:
        st.info("Belum ada data untuk grafik harian.")
        return

    daily = (
        frame.groupby("tanggal", as_index=False)
        .size()
        .rename(columns={"size": "jumlah"})
    )
    daily["tanggal"] = pd.to_datetime(daily["tanggal"])

    chart = (
        alt.Chart(daily)
        .mark_bar(size=18, cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color=color)
        .encode(
            x=alt.X("tanggal:T", title=None, axis=alt.Axis(format="%d %b", grid=False)),
            y=alt.Y(
                "jumlah:Q",
                title="Aduan",
                axis=alt.Axis(tickMinStep=1, grid=True),
            ),
            tooltip=[
                alt.Tooltip("tanggal:T", title="Tanggal", format="%d %B %Y"),
                alt.Tooltip("jumlah:Q", title="Jumlah aduan"),
            ],
        )
        .properties(height=260)
    )

    st.altair_chart(chart, width="stretch")


def render_kategori_chart(frame, color):
    if frame.empty:
        st.info("Belum ada data untuk grafik kategori.")
        return

    kategori_tampil = frame["kategori_perkiraan"].replace("", "Belum terklasifikasi")
    kategori = (
        frame.assign(kategori_tampil=kategori_tampil)
        .groupby("kategori_tampil", as_index=False)
        .size()
        .rename(columns={"size": "jumlah"})
        .sort_values("jumlah", ascending=False)
        .head(TOP_KATEGORI)
    )

    chart = (
        alt.Chart(kategori)
        .mark_bar(size=16, cornerRadiusTopRight=4, cornerRadiusBottomRight=4, color=color)
        .encode(
            y=alt.Y("kategori_tampil:N", title=None, sort="-x"),
            x=alt.X(
                "jumlah:Q",
                title="Aduan",
                axis=alt.Axis(tickMinStep=1),
            ),
            tooltip=[
                alt.Tooltip("kategori_tampil:N", title="Kategori (perkiraan)"),
                alt.Tooltip("jumlah:Q", title="Jumlah"),
            ],
        )
        .properties(height=max(160, 32 * len(kategori)))
    )

    st.altair_chart(chart, width="stretch")


def render_detail(frame, state, show_real_id):
    section_heading("search", "Detail aduan")

    if frame.empty:
        st.info("Tidak ada aduan yang cocok dengan filter.")
        return

    labels = {
        f"{row.kode} — {short(row.masalah, 50)} ({row.waktu_masuk:%d %b %Y %H:%M})": row.kode
        for row in frame.itertuples()
    }
    picked_label = st.selectbox("Pilih kode aduan", list(labels))
    kode = labels[picked_label]
    record = frame[frame["kode"] == kode].iloc[0]

    with st.container(border=True):
        left, right = st.columns([1, 1])

        with left:
            st.markdown("**Isi aduan**")
            chat_id = record["chat_id"] if show_real_id else mask_chat_id(record["chat_id"])
            st.markdown(
                f"- **Kode**: `{record['kode']}`\n"
                f"- **Masuk**: {record['waktu_masuk']:%d %B %Y %H:%M} WIB\n"
                f"- **Masalah**: {record['masalah'] or '-'}\n"
                f"- **Lokasi**: {record['lokasi'] or '-'}\n"
                f"- **Waktu kejadian**: {record['waktu_kejadian'] or '-'}\n"
                f"- **Foto**: {'ada' if record['has_foto'] else 'tidak ada'}\n"
                f"- **Chat ID**: `{chat_id}`"
            )

            foto_path = record.get("foto_path", "")
            if foto_path and Path(foto_path).exists():
                st.image(foto_path, caption=f"Bukti foto aduan {record['kode']}", width="stretch")
            elif record["has_foto"]:
                st.caption("Foto tercatat ada, tapi file lokalnya tidak ditemukan (aduan lama sebelum fitur simpan-foto aktif).")

        session = state.get("session_codes", {}).get(kode, {})
        context = session.get("context", {}) if isinstance(session, dict) else {}

        with right:
            st.markdown(
                f'**Perkiraan otomatis** {icon_img("help-circle", size="0.9em")}',
                unsafe_allow_html=True,
            )
            st.caption("Diisi lewat kemiripan ke data historis SP4N-LAPOR! -- bukan klasifikasi resmi.")
            st.markdown(
                f"- **Kategori**: {record.get('kategori_perkiraan') or '-'}\n"
                f"- **Instansi**: {record.get('instansi_perkiraan') or '-'}\n"
                f"- **Kecamatan**: {record.get('kecamatan_perkiraan') or '-'}"
            )
            st.divider()
            st.markdown("**Konteks sesi**")
            if context:
                st.markdown(f"_Pertanyaan_: {context.get('question', '-')}")
                st.markdown(f"_Jawaban bot_: {short(context.get('answer', '-'), 600)}")
            else:
                st.caption("Tidak ada konteks sesi tersimpan untuk kode ini.")

        history = state.get("chat_histories", {}).get(str(record["chat_id"]), [])
        if history:
            with st.expander(f"Riwayat percakapan chat ini ({len(history)} pesan)"):
                for message in history[-30:]:
                    role = "Warga" if message.get("role") == "user" else "Bot"
                    st.markdown(f"**{role}**: {short(message.get('content', ''), 800)}")
        else:
            st.caption("Riwayat percakapan tidak tersedia.")


@st.cache_data(show_spinner="Memuat basis pengetahuan ke memori (sekali saja)...")
def load_corpus(generation):
    """Baca seluruh korpus markdown sekali, simpan di memori (~31 MB).

    Pencarian berikutnya jadi hitungan detik, bukan rescan disk seperti CLI.
    Parameter `generation` hanya cache key -- dinaikkan oleh tombol muat ulang.
    """
    documents = []

    for corpus_dir in find_corpus_dirs(None):
        for path in sorted(corpus_dir.rglob("*.md")):
            try:
                raw_text = decode_markdown(path.read_bytes())
            except OSError:
                continue

            if raw_text is None:
                continue

            documents.append(
                {
                    "path": path,
                    "corpus": corpus_dir.name,
                    "text": raw_text,
                    "lower": raw_text.lower(),
                }
            )

    return documents


def search_corpus_in_memory(documents, corpora, pattern, options):
    prefilter = options["prefilter"]
    if isinstance(prefilter, bytes):
        prefilter = prefilter.decode("utf-8")

    hits = []
    scanned = 0

    for document in documents:
        if corpora and document["corpus"] not in corpora:
            continue

        scanned += 1
        if prefilter and prefilter not in document["lower"]:
            continue

        hit = scan_text(document["text"], document["path"], pattern, options)
        if hit:
            hits.append(hit)

    return hits, scanned


def render_match_line(match):
    """Snippet dengan bagian yang cocok di-highlight; semua teks di-escape dulu
    supaya isi korpus tidak pernah diperlakukan sebagai HTML."""
    snippet = (
        f"{html.escape(match['before'])}"
        f"<mark>{html.escape(match['match'])}</mark>"
        f"{html.escape(match['after'])}"
    )
    st.markdown(
        f"<div style='font-size:0.86rem;line-height:1.55;margin:0.15rem 0'>"
        f"<code>L{match['line']}</code> {snippet}</div>",
        unsafe_allow_html=True,
    )


_STOPWORDS_ID = {
    "yang", "dan", "atau", "di", "ke", "dari", "dapat", "kepada", "adalah",
    "pada", "untuk", "dengan", "oleh", "ini", "itu", "serta", "dalam", "akan",
    "juga", "tidak", "bagi", "atas", "para", "sebagai", "tersebut", "secara",
    "tanpa", "karena", "jika", "kalau", "namun", "tapi", "agar", "supaya",
    "antara", "sudah", "belum", "masih", "saja", "pun", "lain", "hal", "cara",
    "apa", "siapa", "bagaimana", "kenapa", "mengapa", "dimana", "kapan",
    "sebuah", "suatu", "beberapa", "semua", "setiap", "adanya", "berupa",
    "bahwa", "yaitu", "yakni", "maupun", "lainnya", "sesuai", "menurut",
}


def extract_keywords(text, min_length=3):
    """Pecah teks bebas jadi daftar kata penting (lowercase, dedup, kata
    fungsi & kata pendek dibuang). Dipakai mode "Cocokkan kata kunci": teks
    yang ditempel (jawaban model, sudah diparafrase) tidak akan pernah cocok
    persis kata-demi-kata dengan redaksi asli di korpus, jadi pencarian
    literal (satu string utuh) tidak cocok untuk kasus ini -- lihat
    render_corpus_search()."""
    words = re.findall(r"[a-zA-Z0-9]+(?:[/\-][a-zA-Z0-9]+)*", text.lower())
    seen = set()
    keywords = []
    for word in words:
        if len(word) < min_length or word in _STOPWORDS_ID or word in seen:
            continue
        seen.add(word)
        keywords.append(word)
    return keywords


def search_corpus_by_keywords(documents, corpora, keywords, max_results=20):
    """Ranking dokumen berdasarkan JUMLAH kata kunci yang cocok (whole-word,
    case-insensitive) -- bukan satu string utuh berurutan seperti mode
    literal. Cocok untuk validasi "apakah jawaban model ini benar berdasar
    korpus", karena jawaban model = parafrase, bukan kutipan persis sumber."""
    if not keywords:
        return []

    patterns = [(kw, re.compile(rf"\b{re.escape(kw)}\b")) for kw in keywords]
    results = []

    for document in documents:
        if corpora and document["corpus"] not in corpora:
            continue

        matched = [kw for kw, pat in patterns if pat.search(document["lower"])]
        if not matched:
            continue

        metadata = read_frontmatter(document["text"])
        results.append(
            {
                "path": document["path"],
                "corpus": document["corpus"],
                "text": document["text"],
                "matched": matched,
                "score": len(matched) / len(keywords),
                "chunk_id": metadata.get("chunk_id", document["path"].stem),
                "jenis": metadata.get("jenis", "-"),
                "tipe_dokumen": metadata.get("tipe_dokumen", "-"),
                "dokumen": metadata.get("dokumen", "-"),
                "section": metadata.get("section", "-"),
            }
        )

    results.sort(key=lambda item: (item["score"], len(item["matched"])), reverse=True)
    return results[:max_results]


def render_keyword_snippet(text, matched_keywords, window=420):
    """Tampilkan cuplikan chunk di sekitar konsentrasi kata kunci tercocok
    tertinggi (bukan selalu awal dokumen), dengan tiap kata kunci di-highlight."""
    body = text[find_body_start(text):] or text
    lower = body.lower()

    positions = []
    for kw in matched_keywords:
        for match in re.finditer(rf"\b{re.escape(kw)}\b", lower):
            positions.append(match.start())
    positions.sort()

    best_start = positions[0] if positions else 0
    if positions:
        best_count, best_idx = 0, 0
        for i, pos in enumerate(positions):
            count = sum(1 for p in positions if pos <= p < pos + window)
            if count > best_count:
                best_count, best_idx = count, i
        best_start = positions[best_idx]

    start = max(0, best_start - 80)
    end = min(len(body), start + window)
    snippet = body[start:end]

    escaped = html.escape(snippet)
    for kw in sorted(matched_keywords, key=len, reverse=True):
        escaped = re.sub(
            rf"\b({re.escape(html.escape(kw))})\b",
            r"<mark>\1</mark>",
            escaped,
            flags=re.IGNORECASE,
        )

    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(body) else ""
    st.markdown(
        f"<div style='font-size:0.88rem;line-height:1.6;margin:0.3rem 0'>"
        f"{prefix}{escaped}{suffix}</div>",
        unsafe_allow_html=True,
    )


def render_corpus_search():
    section_heading("database", "Cari Basis Pengetahuan")
    st.caption(
        "Cari pernyataan/jawaban di seluruh basis pengetahuan untuk memastikan "
        "isinya benar-benar bersumber dari korpus, bukan karangan model."
    )

    if "corpus_generation" not in st.session_state:
        st.session_state.corpus_generation = 0

    documents = load_corpus(st.session_state.corpus_generation)

    if not documents:
        st.warning("Tidak ada file .md yang terbaca di folder corpus_*.")
        return

    corpus_names = sorted({document["corpus"] for document in documents})

    mode = st.radio(
        "Mode pencarian",
        ["Cocokkan kata kunci (rekomendasi)", "Literal / regex"],
        horizontal=True,
        help=(
            "'Cocokkan kata kunci': tempel jawaban model apa adanya (boleh 1 kalimat "
            "atau utuh) -- sistem cari chunk dengan kata kunci penting yang paling "
            "banyak cocok, walau redaksinya sudah diparafrase model. "
            "'Literal / regex': cari satu frasa persis berurutan -- cocok untuk cek "
            "fakta pendek yang sudah Anda tahu redaksi persisnya (mis. 'Pasal 2 ayat (2)')."
        ),
    )
    st.markdown(
        '<div class="mode-caption">Pakai mode kata kunci untuk validasi jawaban model '
        "(parafrase). Pakai mode literal untuk cek frasa/istilah pendek persis.</div>",
        unsafe_allow_html=True,
    )

    if mode.startswith("Cocokkan"):
        render_keyword_search_mode(documents, corpus_names)
    else:
        render_literal_search_mode(documents, corpus_names)


def render_keyword_search_mode(documents, corpus_names):
    with st.form("form_cari_kata_kunci"):
        text = st.text_area(
            "Tempel jawaban model (atau bagian darinya)",
            placeholder=(
                "contoh: Pengaduan dapat disampaikan kepada Kepala Desa/Lurah atau "
                "Camat setempat, atau kepada Kepala Dinas atau pejabat lain yang "
                "ditunjuk oleh Kepala Dinas."
            ),
            height=110,
        )
        corpora = st.multiselect(
            "Basis Pengetahuan",
            corpus_names,
            default=corpus_names,
            help="Kosongkan untuk mencari di semua basis pengetahuan.",
        )
        max_results = st.slider("Maks. chunk ditampilkan", 1, 30, 10)
        submitted = st.form_submit_button("Validasi ke korpus", type="primary")

    if not submitted:
        return

    keywords = extract_keywords(text)
    if not keywords:
        st.warning("Teks terlalu pendek atau cuma berisi kata umum -- tempel kalimat yang lebih lengkap.")
        return

    with st.spinner("Mencocokkan kata kunci ke seluruh korpus..."):
        results = search_corpus_by_keywords(documents, corpora, keywords, max_results=max_results)

    st.markdown(
        '<div class="chip-row">' + "".join(f'<span class="chip">{html.escape(kw)}</span>' for kw in keywords) + "</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"{len(keywords)} kata kunci diekstrak dari teks yang ditempel.")

    if not results:
        st.info(
            "Tidak ada chunk yang cocok dengan kata kunci ini sama sekali -- "
            "kemungkinan jawaban model TIDAK didukung basis pengetahuan. Cek juga di "
            "lihat kode sumber sistem untuk detail implementasi."
        )
        return

    st.success(f"{len(results)} chunk teratas ditemukan, diurutkan dari yang paling cocok.")

    for item in results:
        pct = item["score"] * 100
        badge_class = "score-badge-high" if pct >= 70 else ("score-badge-mid" if pct >= 40 else "score-badge-low")
        with st.container(border=True):
            top_left, top_right = st.columns([3, 1])
            top_left.markdown(f'<div class="hit-card-title">{html.escape(item["dokumen"])}</div>', unsafe_allow_html=True)
            top_right.markdown(
                f'<div style="text-align:right"><span class="score-badge {badge_class}">{pct:.0f}% kata kunci cocok</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="chip-row">'
                f'<span class="chip chip-muted">{html.escape(str(item["jenis"]))}/{html.escape(str(item["tipe_dokumen"]))}</span>'
                f'<span class="chip chip-muted">{html.escape(str(item["section"]))}</span>'
                f'<span class="chip chip-muted">chunk_id: {html.escape(str(item["chunk_id"]))}</span>'
                "</div>",
                unsafe_allow_html=True,
            )
            render_keyword_snippet(item["text"], item["matched"])
            with st.expander(f"Lihat isi chunk lengkap ({len(item['matched'])}/{len(keywords)} kata kunci cocok)"):
                st.text(item["text"])


def render_literal_search_mode(documents, corpus_names):
    with st.form("form_cari_korpus"):
        query = st.text_input(
            "Kata / frasa / regex",
            placeholder='contoh: Mampu mengoperasikan Komputer',
        )

        row1 = st.columns([2, 1, 1, 1])
        corpora = row1[0].multiselect(
            "Basis Pengetahuan",
            corpus_names,
            default=corpus_names,
            help="Kosongkan untuk mencari di semua basis pengetahuan.",
        )
        is_regex = row1[1].checkbox("Regex", help="Perlakukan query sebagai regex Python")
        loose = row1[2].checkbox(
            "Longgar",
            value=True,
            help="Spasi pada query cocok dengan spasi atau baris baru apa pun",
        )
        case_sensitive = row1[3].checkbox("Bedakan huruf besar/kecil")

        row2 = st.columns([1, 1, 1, 1])
        jenis = row2[0].text_input("Filter jenis", placeholder="regulasi / pengaduan")
        tipe_dokumen = row2[1].text_input("Filter tipe_dokumen", placeholder="sop")
        max_results = row2[2].number_input(
            "Maks. file ditampilkan", min_value=1, max_value=200, value=20
        )
        context_chars = row2[3].number_input(
            "Konteks (karakter)", min_value=20, max_value=400, value=90, step=10
        )

        include_frontmatter = st.checkbox(
            "Ikut cari di dalam frontmatter",
            help="Default hanya isi dokumen, supaya metadata tidak ikut mengotori hasil.",
        )
        submitted = st.form_submit_button("Cari", type="primary")

    reload_column, info_column = st.columns([1, 3])
    if reload_column.button("Muat ulang basis pengetahuan"):
        st.session_state.corpus_generation += 1
        st.rerun()
    info_column.caption(
        f"{len(documents)} file .md termuat di memori dari {len(corpus_names)} basis pengetahuan. "
        "Klik muat ulang setelah basis pengetahuan di-rebuild."
    )

    if not submitted:
        return

    if not query.strip():
        st.warning("Isi dulu kata atau frasa yang ingin dicari.")
        return

    try:
        pattern, options = build_options(
            query,
            is_regex=is_regex,
            loose=loose,
            case_sensitive=case_sensitive,
            include_frontmatter=include_frontmatter,
            context_chars=int(context_chars),
            filters={
                "jenis": jenis or None,
                "tipe_dokumen": tipe_dokumen or None,
            },
        )
    except SystemExit as error:
        st.error(str(error))
        return

    with st.spinner("Mencari..."):
        hits, scanned = search_corpus_in_memory(documents, corpora, pattern, options)

    st.success(f"{len(hits)} file cocok dari {scanned} file yang dipindai.")

    if not hits:
        st.info(
            "Tidak ada yang cocok. Coba kurangi kata, aktifkan 'Longgar', "
            "atau periksa filter jenis/tipe_dokumen."
        )
        return

    for hit in hits[: int(max_results)]:
        with st.container(border=True):
            top_left, top_right = st.columns([3, 1])
            top_left.markdown(f'<div class="hit-card-title">{html.escape(hit["dokumen"])}</div>', unsafe_allow_html=True)
            top_right.markdown(
                f'<div style="text-align:right"><span class="chip chip-muted">{hit["total_matches"]} kecocokan</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="chip-row">'
                f'<span class="chip chip-muted">{html.escape(str(hit["jenis"]))}/{html.escape(str(hit["tipe_dokumen"]))}</span>'
                f'<span class="chip chip-muted">{html.escape(str(hit["section"]))}</span>'
                f'<span class="chip chip-muted">chunk_id: {html.escape(str(hit["chunk_id"]))}</span>'
                f'<span class="chip chip-muted">halaman: {html.escape(str(hit["halaman"]))}</span>'
                "</div>",
                unsafe_allow_html=True,
            )
            st.caption(hit["relative_path"])

            for match in hit["matches"]:
                render_match_line(match)

            with st.expander("Lihat isi chunk lengkap"):
                st.text(Path(hit["path"]).read_text(encoding="utf-8"))

    if len(hits) > int(max_results):
        st.caption(
            f"{len(hits) - int(max_results)} file lain disembunyikan. "
            "Naikkan 'Maks. file ditampilkan'."
        )


def load_verification_log():
    """Catatan verifikasi manual (benar/salah + catatan) per jawaban bot.

    Data verifikasi disimpan terpisah dari data aduan (read-only dari dashboard ini) --
    file JSON kecil khusus untuk mencatat hasil pengecekan manual, supaya
    tetap ada walau riwayat chat di state.json ikut ter-trim (MAX_HISTORY_MESSAGES)."""
    if not VERIFICATION_LOG_PATH.exists():
        return {}
    try:
        return json.loads(VERIFICATION_LOG_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_verification_log(log):
    VERIFICATION_LOG_PATH.write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def verification_key(chat_id, turn_index, answer):
    """Key stabil per (chat, urutan pesan, isi jawaban) -- tidak berubah selama
    riwayat pesan itu belum ter-trim dari state.json."""
    raw = f"{chat_id}|{turn_index}|{answer[:200]}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def flatten_qa_turns(state):
    """Ubah chat_histories (per chat_id) jadi daftar giliran tanya-jawab datar,
    dipasangkan dengan pertanyaan pengguna sebelumnya (kalau ada)."""
    turns = []
    histories = state.get("chat_histories", {})

    for chat_id, history in histories.items():
        chat_turns = []
        for index, message in enumerate(history):
            if message.get("role") != "assistant":
                continue

            question = ""
            if index > 0 and history[index - 1].get("role") == "user":
                question = history[index - 1].get("content", "")

            chat_turns.append(
                {
                    "chat_id": chat_id,
                    "turn_index": index,
                    "question": question,
                    "answer": message.get("content", ""),
                    "mode": message.get("mode") or "-",
                    "route_mode": message.get("route_mode") or "-",
                    "resolved_jenis": message.get("resolved_jenis") or "-",
                    "sources": message.get("sources") or [],
                }
            )
        turns.extend(reversed(chat_turns))

    return turns


def render_source_evidence(sources, key_prefix):
    if not sources:
        st.caption(
            "Tidak ada sumber terlampir -- jawaban ini kemungkinan dijawab "
            "langsung oleh logika backend (template/aturan/penolakan), bukan "
            "hasil retrieval + generation lewat Qwen."
        )
        return

    for idx, source in enumerate(sources):
        metadata = source.get("metadata", {}) or {}
        chunk_id = metadata.get("chunk_id", source.get("chunk_id", "-"))
        base_score = source.get("base_score", source.get("score"))
        rerank_score = source.get("rerank_score", source.get("score"))
        path = metadata.get("path")

        with st.container(border=True):
            top_left, top_right = st.columns([3, 1])
            top_left.markdown(f'<div class="hit-card-title">{html.escape(str(metadata.get("dokumen", "-")))}</div>', unsafe_allow_html=True)
            score_chips = ""
            if base_score is not None:
                score_chips += f'<span class="chip chip-muted">base {base_score:.3f}</span>'
            if rerank_score is not None:
                score_chips += f'<span class="chip chip-muted">final {rerank_score:.3f}</span>'
            top_right.markdown(f'<div style="text-align:right">{score_chips or "-"}</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="chip-row">'
                f'<span class="chip chip-muted">{html.escape(str(metadata.get("section", "-")))}</span>'
                f'<span class="chip chip-muted">chunk_id: {html.escape(str(chunk_id))}</span>'
                "</div>",
                unsafe_allow_html=True,
            )

            if path and Path(path).exists():
                show_key = f"{key_prefix}_src_{idx}"
                if st.checkbox("Lihat isi chunk lengkap (bukti dari basis pengetahuan)", key=show_key):
                    st.text(Path(path).read_text(encoding="utf-8"))
            elif path:
                st.caption(f"File sumber tidak ditemukan di path tersimpan: `{path}` (basis pengetahuan mungkin sudah di-rebuild/dipindah).")


def render_verification_controls(log, key, turn):
    existing = log.get(key)

    if existing:
        if existing.get("status") == "ok":
            badge = f'{icon_img("circle-check", size="0.95em")}Sesuai sumber'
        else:
            badge = f'{icon_img("alert-circle", size="0.95em")}Perlu revisi'
        st.markdown(
            f"**Status verifikasi:** {badge} · dicatat {existing.get('verified_at', '-')}",
            unsafe_allow_html=True,
        )
        if existing.get("catatan"):
            st.caption(f"Catatan: {existing['catatan']}")
        if st.button("Hapus tanda verifikasi", key=f"hapus_{key}"):
            log.pop(key, None)
            save_verification_log(log)
            st.rerun()
        return

    with st.form(key=f"form_verif_{key}"):
        status = st.radio(
            "Hasil pengecekan manual",
            ["Sesuai sumber", "Tidak sesuai / perlu revisi"],
            horizontal=True,
            key=f"status_{key}",
        )
        catatan = st.text_area(
            "Catatan (opsional)",
            placeholder="Mis. jawaban cocok dengan Pasal 6 ayat 2, chunk_id ... memang berisi itu.",
            key=f"catatan_{key}",
        )
        submitted = st.form_submit_button("Simpan verifikasi")

    if submitted:
        log[key] = {
            "status": "ok" if status == "Sesuai sumber" else "revisi",
            "catatan": catatan.strip(),
            "verified_at": datetime.now(WIB).strftime("%d %b %Y %H:%M"),
            "chat_id": turn["chat_id"],
            "question": turn["question"],
        }
        save_verification_log(log)
        st.rerun()


def render_answer_verification(state):
    section_heading("shield-check", "Verifikasi Jawaban AI")
    st.caption(
        "Cek jawaban bot terhadap sumber aslinya di basis pengetahuan -- untuk "
        "persiapan menjawab pertanyaan penguji seperti 'atas dasar apa jawaban ini "
        "benar?'."
    )

    all_turns = flatten_qa_turns(state)
    verif_log = load_verification_log()

    if not all_turns:
        st.info("Belum ada riwayat tanya-jawab untuk ditampilkan.")
        return

    verified_ok = sum(1 for t in all_turns if verif_log.get(verification_key(t["chat_id"], t["turn_index"], t["answer"]), {}).get("status") == "ok")
    verified_revisi = sum(1 for t in all_turns if verif_log.get(verification_key(t["chat_id"], t["turn_index"], t["answer"]), {}).get("status") == "revisi")
    belum = len(all_turns) - verified_ok - verified_revisi

    tile_cols = st.columns(4)
    with tile_cols[0]:
        metric_card("report", "Total jawaban tercatat", len(all_turns))
    with tile_cols[1]:
        metric_card("circle-check", "Sudah diverifikasi sesuai", verified_ok)
    with tile_cols[2]:
        metric_card("alert-circle", "Ditandai perlu revisi", verified_revisi)
    with tile_cols[3]:
        metric_card("clock", "Belum diverifikasi", belum)

    with st.form("form_filter_verifikasi"):
        keyword = st.text_input("Cari kata di pertanyaan / jawaban", placeholder="mis. Pasal 6, disposisi, jalan berlubang")

        row = st.columns([1, 1, 1])
        modes_available = sorted({t["mode"] for t in all_turns})
        mode_filter = row[0].selectbox("Filter mode", ["Semua"] + modes_available)
        only_with_sources = row[1].checkbox("Hanya yang punya sumber", value=False)
        only_unverified = row[2].checkbox("Hanya yang belum diverifikasi", value=False)

        max_results = st.number_input("Maks. ditampilkan", min_value=1, max_value=100, value=15)
        submitted = st.form_submit_button("Cari", type="primary")

    if not submitted:
        return

    filtered = all_turns
    if keyword.strip():
        needle = keyword.strip().lower()
        filtered = [
            t for t in filtered
            if needle in t["question"].lower() or needle in t["answer"].lower()
        ]
    if mode_filter != "Semua":
        filtered = [t for t in filtered if t["mode"] == mode_filter]
    if only_with_sources:
        filtered = [t for t in filtered if t["sources"]]
    if only_unverified:
        filtered = [
            t for t in filtered
            if verification_key(t["chat_id"], t["turn_index"], t["answer"]) not in verif_log
        ]

    st.success(f"{len(filtered)} jawaban cocok dari {len(all_turns)} total.")

    for turn in filtered[: int(max_results)]:
        key = verification_key(turn["chat_id"], turn["turn_index"], turn["answer"])
        with st.container(border=True):
            st.markdown(
                '<div class="chip-row">'
                f'<span class="chip chip-muted">chat {html.escape(mask_chat_id(turn["chat_id"]))}</span>'
                f'<span class="chip chip-muted">mode: {html.escape(str(turn["mode"]))}</span>'
                f'<span class="chip chip-muted">route: {html.escape(str(turn["route_mode"]))}</span>'
                f'<span class="chip chip-muted">jenis: {html.escape(str(turn["resolved_jenis"]))}</span>'
                "</div>",
                unsafe_allow_html=True,
            )
            if turn["question"]:
                st.markdown(f"**Pertanyaan:** {turn['question']}")
            st.markdown(f"**Jawaban AI:** {turn['answer']}")

            with st.expander(f"Lihat bukti/sumber ({len(turn['sources'])})"):
                render_source_evidence(turn["sources"], key)

            render_verification_controls(verif_log, key, turn)

    if len(filtered) > int(max_results):
        st.caption(f"{len(filtered) - int(max_results)} jawaban lain disembunyikan. Naikkan 'Maks. ditampilkan'.")


def render_monitoring():
    st.caption(
        "Monitoring aduan masuk dari bot Telegram. Halaman ini hanya menampilkan data "
        "dan tidak pernah mengubah data aduan."
    )

    db_signature = file_signature(db.DB_PATH)
    frame = load_complaints(db_signature)
    state = load_state(db_signature)

    _, header_right = st.columns([4, 1])
    with header_right:
        if st.button("Muat ulang data", width="stretch"):
            load_complaints.clear()
            load_state.clear()
            st.rerun()

    if frame.empty:
        st.warning(
            "Belum ada aduan masuk. "
            "Dashboard akan terisi otomatis setelah ada aduan masuk lewat bot Telegram."
        )
        return

    with st.sidebar:
        sidebar_section("Filter · Monitoring Aduan")

        min_date = frame["tanggal"].min()
        max_date = frame["tanggal"].max()
        date_range = st.date_input(
            "Rentang tanggal masuk",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        keyword = st.text_input(
            "Kata kunci",
            placeholder="cari di masalah / lokasi / kode",
        )
        foto_filter = st.selectbox(
            "Lampiran foto",
            ["Semua", "Ada foto", "Tanpa foto"],
        )

        sidebar_section("Privasi")
        show_real_id = st.checkbox(
            "Tampilkan chat_id asli",
            value=False,
            help=(
                "chat_id adalah identitas pelapor. Default disamarkan; "
                "aktifkan hanya bila memang perlu untuk tindak lanjut."
            ),
        )

    filtered = frame

    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered = filtered[
            (filtered["tanggal"] >= start_date) & (filtered["tanggal"] <= end_date)
        ]

    if keyword:
        needle = keyword.strip().lower()
        haystack = (
            filtered["masalah"].str.lower()
            + " "
            + filtered["lokasi"].str.lower()
            + " "
            + filtered["waktu_kejadian"].str.lower()
            + " "
            + filtered["kode"].str.lower()
        )
        filtered = filtered[haystack.str.contains(needle, regex=False, na=False)]

    if foto_filter == "Ada foto":
        filtered = filtered[filtered["has_foto"]]
    elif foto_filter == "Tanpa foto":
        filtered = filtered[~filtered["has_foto"]]

    render_stat_tiles(filtered, frame)
    st.divider()

    color = accent_color()
    chart_left, chart_right = st.columns(2)

    with chart_left:
        with st.container(border=True):
            st.markdown(
                f'**{icon_img("trending-up", size="0.95em")}Aduan masuk per hari**',
                unsafe_allow_html=True,
            )
            render_daily_chart(filtered, color)

    with chart_right:
        with st.container(border=True):
            st.markdown(
                f'**{icon_img("tag", size="0.95em")}Kategori terbanyak (top {TOP_KATEGORI}, perkiraan)**',
                unsafe_allow_html=True,
            )
            render_kategori_chart(filtered, color)

    st.divider()
    st.subheader(f"Daftar aduan ({len(filtered)})")

    table = filtered.copy()
    table["chat_id_tampil"] = (
        table["chat_id"] if show_real_id else table["chat_id"].map(mask_chat_id)
    )
    table = table[
        [
            "kode",
            "waktu_masuk",
            "masalah",
            "kategori_perkiraan",
            "kecamatan_perkiraan",
            "lokasi",
            "waktu_kejadian",
            "has_foto",
            "chat_id_tampil",
        ]
    ]

    st.caption(
        "Kolom **(perkiraan)** diisi otomatis lewat kemiripan ke data historis "
        "SP4N-LAPOR! -- bukan klasifikasi resmi Diskominfo."
    )
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            "kode": st.column_config.TextColumn("Kode", width="small"),
            "waktu_masuk": st.column_config.DatetimeColumn(
                "Masuk (WIB)",
                format="DD MMM YYYY HH:mm",
                width="medium",
            ),
            "masalah": st.column_config.TextColumn("Masalah", width="large"),
            "kategori_perkiraan": st.column_config.TextColumn("Kategori (perkiraan)", width="medium"),
            "kecamatan_perkiraan": st.column_config.TextColumn("Kecamatan (perkiraan)", width="small"),
            "lokasi": st.column_config.TextColumn("Lokasi", width="medium"),
            "waktu_kejadian": st.column_config.TextColumn("Waktu kejadian"),
            "has_foto": st.column_config.CheckboxColumn("Foto", width="small"),
            "chat_id_tampil": st.column_config.TextColumn("Chat ID", width="small"),
        },
    )

    st.download_button(
        "Unduh CSV (sesuai filter)",
        data=table.to_csv(index=False).encode("utf-8-sig"),
        file_name="aduan_telegram.csv",
        mime="text/csv",
    )

    st.divider()
    render_detail(filtered, state, show_real_id)


def render_kelola_pipeline():
    """Tab: Kelola Basis Pengetahuan — upload dataset, rebuild corpus & index."""

    section_heading("settings-automation", "Kelola Basis Pengetahuan")

    stats = pipeline_manager.get_index_stats()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("file-spreadsheet", "File Dataset", len(stats["dataset_files"]))
    with col2:
        metric_card("table", "Baris CSV", f'{stats["csv_rows"]:,}' if stats["csv_exists"] else "—")
    with col3:
        metric_card("files", "Chunk Aduan", f'{stats["corpus_aduan_chunks"]:,}')
    with col4:
        metric_card("database", "Total Chunk", f'{stats["total_chunks"]:,}')

    col5, col6, col7 = st.columns(3)
    with col5:
        metric_card("file-text", "Chunk SOP", f'{stats["corpus_sop_chunks"]:,}')
    with col6:
        chroma_status = "Tersedia" if stats["chroma_exists"] else "Belum ada"
        metric_card("vector", "Index ChromaDB", chroma_status)
    with col7:
        bm25_status = f'{stats["bm25_size_mb"]} MB' if stats["bm25_exists"] else "Belum ada"
        metric_card("search", "Index BM25", bm25_status)

    st.divider()

    section_heading("file-spreadsheet", "Dataset Excel")

    dataset_files = stats["dataset_files"]
    if dataset_files:
        st.markdown(
            "File dataset saat ini: "
            + ", ".join(f"`{f}`" for f in dataset_files)
        )
    else:
        st.info("Belum ada file Excel dataset yang diunggah.")

    uploaded = st.file_uploader(
        "Upload file Excel pengaduan (.xlsx)",
        type=["xlsx"],
        accept_multiple_files=True,
        key="upload_excel",
    )

    if uploaded:
        overwrite = st.checkbox("Timpa file yang sudah ada", value=False, key="overwrite_excel")
        if st.button("Simpan File", key="btn_save_excel", type="primary"):
            results = pipeline_manager.save_uploaded_excel(uploaded, overwrite=overwrite)
            for fname, status in results:
                if status == "tersimpan":
                    st.success(f"{fname}: {status}")
                else:
                    st.warning(f"{fname}: {status}")
            st.rerun()

    st.divider()

    section_heading("file-text", "Dokumen Regulasi / SOP (PDF)")

    regulasi_files = pipeline_manager.list_regulasi_files()
    if regulasi_files:
        st.markdown(
            "File regulasi/SOP saat ini: "
            + ", ".join(f"`{f}`" for f in regulasi_files)
        )
    else:
        st.info(
            "Belum ada dokumen regulasi/SOP. Jika tidak diunggah, "
            "sistem akan menggunakan basis pengetahuan SOP bawaan."
        )

    uploaded_pdf = st.file_uploader(
        "Upload file PDF regulasi/SOP",
        type=["pdf"],
        accept_multiple_files=True,
        key="upload_pdf",
    )

    if uploaded_pdf:
        overwrite_pdf = st.checkbox("Timpa file yang sudah ada", value=False, key="overwrite_pdf")
        if st.button("Simpan PDF", key="btn_save_pdf", type="primary"):
            results = pipeline_manager.save_uploaded_pdfs(uploaded_pdf, overwrite=overwrite_pdf)
            for fname, status in results:
                if status == "tersimpan":
                    st.success(f"{fname}: {status}")
                else:
                    st.warning(f"{fname}: {status}")
            st.rerun()

    st.divider()

    section_heading("player-play", "Jalankan Pipeline")

    st.markdown(
        "Pipeline akan menjalankan 5 tahap secara berurutan:\n"
        "1. **Pembersihan Data** — Membersihkan dan menyiapkan data Excel\n"
        "2. **Bangun Korpus Aduan** — Memotong data aduan menjadi bagian-bagian kecil (dengan anonimisasi)\n"
        "3. **Bangun Korpus Regulasi/SOP** — Mengekstrak teks dari dokumen PDF yang diunggah, "
        "atau menggunakan korpus SOP bawaan jika tidak ada PDF.\n"
        "4. **Bangun Indeks Vektor** — Mengubah teks menjadi representasi numerik untuk pencarian semantik\n"
        "5. **Bangun Indeks Kata Kunci** — Membuat indeks pencarian berdasarkan kata kunci"
    )

    st.info(
        "Proses ini bisa memakan waktu beberapa menit, tergantung jumlah data. "
        "Pastikan koneksi internet stabil selama proses berjalan.",
        icon="ℹ️",
    )

    if st.button("Jalankan Seluruh Pipeline", key="btn_run_pipeline", type="primary"):
        progress_bar = st.progress(0, text="Memulai pipeline...")
        log_container = st.container()
        total_stages = len(pipeline_manager.STAGES)

        results = []
        for i, (name, fn) in enumerate(pipeline_manager.STAGES):
            progress_bar.progress(
                (i) / total_stages,
                text=f"Tahap {i+1}/{total_stages}: {name}...",
            )
            import time as _time
            t0 = _time.time()
            try:
                log = fn()
                status = "ok"
            except Exception as exc:
                log = f"ERROR: {exc}"
                status = "error"
            elapsed = _time.time() - t0
            results.append((name, status, log, elapsed))

            with log_container:
                if status == "ok":
                    with st.expander(f"✅ {name} ({elapsed:.1f}s)", expanded=False):
                        st.code(log, language="text")
                else:
                    with st.expander(f"❌ {name} ({elapsed:.1f}s)", expanded=True):
                        st.code(log, language="text")

            if status == "error":
                break

        all_ok = all(s == "ok" for _, s, _, _ in results)
        if all_ok:
            progress_bar.progress(1.0, text="Pipeline selesai.")
            st.success("Seluruh pipeline berhasil dijalankan.")
        else:
            failed = next(n for n, s, _, _ in results if s == "error")
            progress_bar.progress(1.0, text=f"Pipeline gagal di tahap: {failed}")
            st.error(f"Pipeline berhenti karena error di tahap: {failed}")

    st.divider()

    section_heading("list-check", "Jalankan Tahap Tertentu")

    stage_names = [name for name, _ in pipeline_manager.STAGES]
    selected_stage = st.selectbox(
        "Pilih tahap",
        options=stage_names,
        key="select_stage",
    )

    if st.button("Jalankan Tahap Ini", key="btn_run_stage"):
        stage_fn = dict(pipeline_manager.STAGES)[selected_stage]
        with st.spinner(f"Menjalankan: {selected_stage}..."):
            import time as _time
            t0 = _time.time()
            try:
                log = stage_fn()
                elapsed = _time.time() - t0
                st.success(f"{selected_stage} selesai ({elapsed:.1f}s)")
                with st.expander("Log", expanded=True):
                    st.code(log, language="text")
            except Exception as exc:
                elapsed = _time.time() - t0
                st.error(f"{selected_stage} gagal ({elapsed:.1f}s): {exc}")

    st.divider()

    render_kelola_bot()


def render_kelola_bot():
    """Section: Start/stop Telegram bot dari dashboard."""
    section_heading("brand-telegram", "Telegram Bot")

    running, info = pipeline_manager.bot_status()

    if running:
        st.success(f"🟢 Bot {info}")
        if st.button("Hentikan Bot", key="btn_stop_bot", type="secondary"):
            ok, msg = pipeline_manager.stop_bot()
            if ok:
                st.success(msg)
            else:
                st.error(msg)
            st.rerun()
    else:
        st.info(f"🔴 Bot: {info}")
        if st.button("Jalankan Bot", key="btn_start_bot", type="primary"):
            ok, msg = pipeline_manager.start_bot()
            if ok:
                st.success(msg)
            else:
                st.error(msg)
            st.rerun()


def main():
    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)

    with st.sidebar:
        render_sidebar_brand()

    st.title("Dashboard Admin — RAG Pengaduan Kabupaten Bandung")
    tagline = "Monitoring aduan, penelusuran basis pengetahuan, dan pengelolaan pipeline."
    st.markdown(
        f'<p class="app-tagline">{tagline}</p>',
        unsafe_allow_html=True,
    )

    tab_labels = ["Monitoring Aduan", "Cari Basis Pengetahuan"]
    if SHOW_VERIFIKASI_TAB:
        tab_labels.append("Verifikasi Jawaban AI")
    tab_labels.append("Kelola Basis Pengetahuan")

    tabs = st.tabs(tab_labels)
    tab_monitoring, tab_korpus = tabs[0], tabs[1]

    with tab_monitoring:
        render_monitoring()

    with tab_korpus:
        render_corpus_search()

    tab_idx = 2
    if SHOW_VERIFIKASI_TAB:
        with tabs[tab_idx]:
            db_signature = file_signature(db.DB_PATH)
            state = load_state(db_signature)
            render_answer_verification(state)
        tab_idx += 1

    with tabs[tab_idx]:
        render_kelola_pipeline()

    with st.sidebar:
        st.markdown(
            '<div class="sidebar-footer">RAG Pengaduan &middot; Kabupaten Bandung'
            '<br/>Data lokal, tidak pernah dikirim keluar.</div>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
