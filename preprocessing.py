import pandas as pd
import re
from pathlib import Path


FILES = [
    "dataset_2020.xlsx",
    "dataset_2021.xlsx",
    "dataset_2022.xlsx",
    "dataset_2023.xlsx",
    "dataset_2024.xlsx",
    "dataset_2025.xlsx",
]

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

HEADER_ROW = 18

OUTPUT_CSV = BASE_DIR / "dataset_clean.csv"

DROP_EMPTY_TINDAK_LANJUT = True


def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    text = re.sub(r"http\S+|www\S+", "", text, flags=re.IGNORECASE)

    text = re.sub(r"[^\w\s.,:/()\[\]\-]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def filter_spam_entries(df, min_length=15):

    spam_pattern = re.compile(
        r"^(tes|test|coba|trial|xxx+|\.+|-+)\s*\d*$",
        re.IGNORECASE
    )

    def is_spam(text):
        text = str(text).strip()
        if len(text) < min_length:
            return True
        if spam_pattern.match(text):
            return True
        return False

    before = len(df)
    df = df[~df["isi_laporan"].apply(is_spam)]
    after = len(df)

    print(f"\n[INFO] Entri spam/kotor dibuang: {before - after}")

    return df


def lengkapi_kecamatan(df, fill_value="TIDAK DIKETAHUI"):

    kec_valid = (
        df["kecamatan"]
        .dropna()
        .astype(str)
        .str.upper()
        .str.strip()
        .unique()
        .tolist()
    )

    kec_valid.sort(key=len, reverse=True)

    patterns = [
        (kec, re.compile(r"\b" + re.escape(kec) + r"\b"))
        for kec in kec_valid
    ]

    def detect(row):
        text = f"{row['judul']} {row['isi_laporan']}".upper()
        for kec, pat in patterns:
            if pat.search(text):
                return kec
        return None

    mask = df["kecamatan"].isna()

    terisi = df.loc[mask].apply(detect, axis=1)

    n_terisi = terisi.notna().sum()
    df.loc[mask, "kecamatan"] = terisi

    sisa_kosong = df["kecamatan"].isna().sum()
    df["kecamatan"] = df["kecamatan"].fillna(fill_value)

    print(
        f"\n[INFO] Kecamatan dilengkapi dari teks: {n_terisi}, "
        f"sisa diberi label '{fill_value}': {sisa_kosong}"
    )

    return df


def normalize_columns(df):

    df = df.loc[
        :,
        ~df.columns.astype(str).str.contains("^Unnamed")
    ]

    df.columns = [
        str(c).strip().lower()
        for c in df.columns
    ]

    df = df.loc[
        :,
        ~df.columns.str.match(r"^.+\.\d+$")
    ]

    return df


def merge_tindak_lanjut(df):

    tl_cols = [
        c for c in df.columns
        if c.startswith("tindak lanjut laporan")
    ]

    tl_cols.sort(
        key=lambda c: int(c.split()[-1])
    )

    if not tl_cols:
        return df

    def combine_row(row):

        parts = []

        for i, val in enumerate(row, start=1):

            if pd.notna(val) and str(val).strip():

                parts.append(
                    f"[TINDAK LANJUT {i}] {str(val).strip()}"
                )

        return "\n".join(parts) if parts else ""

    df["tindak_lanjut"] = df[tl_cols].apply(
        combine_row,
        axis=1
    )

    df = df.drop(columns=tl_cols)

    return df


def map_columns(df):

    if "status laporan" in df.columns and "status" in df.columns:
        df = df.drop(columns=["status"])

    col_map = {

        "tracking id": "id",

        "judul laporan": "judul",
        "isi laporan akhir": "isi_laporan",

        "instansi terdisposisi": "instansi",
        "kecamatan": "kecamatan",
        "kategori laporan": "kategori",
        "klasifikasi laporan": "klasifikasi",
        "status laporan": "status",

        "tanggal laporan masuk": "tanggal_masuk",
        "tanggal tindak lanjut pertama": "tanggal_tindak_lanjut_pertama",
        "tanggal laporan selesai": "tanggal_selesai",

    }

    rename_dict = {
        k: v
        for k, v in col_map.items()
        if k in df.columns
    }

    df = df.rename(columns=rename_dict)

    return df


def validate_dataset(df):

    print("\n" + "="*60)
    print("VALIDATION")
    print("="*60)

    print("\n[INFO] Columns:")
    print(df.columns.tolist())

    print("\n[INFO] Missing Values:")
    print(df.isnull().sum())

    label_penanda = [
        "BELUM DIKATEGORIKAN",
        "BELUM DIKLASIFIKASI",
        "TIDAK DIKETAHUI",
    ]

    for kolom, judul in [("kategori", "Kategori"), ("kecamatan", "Kecamatan")]:

        if kolom not in df.columns:
            continue

        isi = df[kolom]
        penanda = isi.isin(label_penanda).sum()

        print(f"\n[INFO] Top {judul}:")
        print(isi[~isi.isin(label_penanda)].value_counts().head(15))

        if penanda:
            print(f"[INFO] {judul} tanpa keterangan: {penanda}")

    print("\n[INFO] Sample Data:")
    print(df.sample(min(3, len(df))))


def tambah_metadata_temporal(df):
    """Parse tanggal level-laporan lalu hitung durasi respon & penyelesaian.

    Granularitas berada pada level siklus hidup laporan (masuk -> tindak
    lanjut pertama -> selesai), BUKAN per entri tindak lanjut, karena hasil
    ekspor SP4N-LAPOR tidak menyediakan stempel waktu untuk tiap tindak
    lanjut. durasi_respon dan durasi_penyelesaian dihitung dalam satuan hari.
    """
    date_cols = [
        "tanggal_masuk",
        "tanggal_tindak_lanjut_pertama",
        "tanggal_selesai",
    ]

    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)

    if "tanggal_masuk" in df.columns and "tanggal_tindak_lanjut_pertama" in df.columns:
        df["durasi_respon"] = (
            df["tanggal_tindak_lanjut_pertama"] - df["tanggal_masuk"]
        ).dt.days

    if "tanggal_masuk" in df.columns and "tanggal_selesai" in df.columns:
        df["durasi_penyelesaian"] = (
            df["tanggal_selesai"] - df["tanggal_masuk"]
        ).dt.days

    for c in date_cols:
        if c in df.columns:
            df[c] = df[c].dt.strftime("%Y-%m-%d").fillna("")

    return df


def main():

    all_df = []


    for file in FILES:

        print(f"\n[LOAD] {file}")

        filepath = DATASET_DIR / file

        df = pd.read_excel(
            filepath,
            header=HEADER_ROW
        )

        unnamed_ratio = (
            df.columns.astype(str).str.contains("^Unnamed").mean()
        )
        if df.empty or unnamed_ratio > 0.5:
            raise ValueError(
                f"{file}: baris header {HEADER_ROW} tampaknya tidak cocok "
                f"({unnamed_ratio:.0%} kolom Unnamed). "
                f"Kolom terbaca: {df.columns.tolist()[:5]}"
            )

        print(f"[LOAD] {file}: {len(df)} baris")

        year_match = re.search(r"(\d{4})", file)
        if year_match:
            df["tahun"] = int(year_match.group(1))

        df = normalize_columns(df)

        df = merge_tindak_lanjut(df)

        df = map_columns(df)

        all_df.append(df)


    df = pd.concat(
        all_df,
        ignore_index=True
    )

    print(f"\n[INFO] Total awal: {len(df)}")


    df = tambah_metadata_temporal(df)


    cols = [
        "id",
        "tahun",
        "judul",
        "isi_laporan",
        "tindak_lanjut",
        "instansi",
        "kecamatan",
        "kategori",
        "klasifikasi",
        "status",
        "tanggal_masuk",
        "tanggal_tindak_lanjut_pertama",
        "tanggal_selesai",
        "durasi_respon",
        "durasi_penyelesaian",
    ]

    df = df[
        [c for c in cols if c in df.columns]
    ]


    if "isi_laporan" not in df.columns:
        raise ValueError(
            "Kolom 'isi_laporan' tidak terbentuk. Cek nama kolom sumber "
            "pada map_columns()."
        )

    df = df.dropna(
        subset=[
            c for c in ["isi_laporan", "tindak_lanjut"]
            if c in df.columns
        ]
    )

    if DROP_EMPTY_TINDAK_LANJUT and "tindak_lanjut" in df.columns:

        before = len(df)

        df = df[df["tindak_lanjut"].astype(str).str.strip() != ""]

        print(
            f"\n[INFO] Laporan tanpa tindak lanjut dibuang: "
            f"{before - len(df)}"
        )


    text_cols = [
        "judul",
        "isi_laporan",
        "tindak_lanjut"
    ]

    for col in text_cols:

        if col in df.columns:

            df[col] = df[col].apply(clean_text)

    df = filter_spam_entries(df)

    df = lengkapi_kecamatan(df)

    if "kategori" in df.columns:
        df["kategori"] = df["kategori"].fillna("BELUM DIKATEGORIKAN")
    if "klasifikasi" in df.columns:
        df["klasifikasi"] = df["klasifikasi"].fillna("BELUM DIKLASIFIKASI")


    before = len(df)

    df = df.drop_duplicates(
        subset=[
            "isi_laporan",
            "tindak_lanjut"
        ]
    )

    after = len(df)

    print(f"\n[INFO] Duplicate removed: {before - after}")


    df = df.reset_index(drop=True)

    print(f"\n[INFO] Total final: {len(df)}")


    validate_dataset(df)


    df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8"
    )

    print(f"\n[DONE] Saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
