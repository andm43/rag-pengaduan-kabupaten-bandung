import re
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "pendukung" / "SOP - SOP Pengaduan.pdf"
OUTPUT_DIR = BASE_DIR / "corpus_regulasi_sop"
OUTPUT_DIR.mkdir(exist_ok=True)

DOC_NAME = "SOP Penyelesaian Pengaduan SP4N-LAPOR"
SOURCE_FILE = PDF_PATH.name

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


SECTIONS = [
    {
        "slug": "identitas",
        "title": "Identitas SOP Penyelesaian Pengaduan SP4N-LAPOR",
        "pages": "1",
        "content": """
SOP ini mengatur penyelesaian pengaduan melalui SP4N-LAPOR di lingkungan Pemerintah Kabupaten Bandung.

Nama SOP: Penyelesaian Pengaduan SP4N-LAPOR.

SOP disahkan oleh Kepala Dinas Komunikasi dan Informatika, Statistik dan Persandian Kabupaten Bandung.

Ruang lingkup SOP mencakup penerimaan, penelaahan, disposisi, tindak lanjut, pemberian jawaban, pemantauan tanggapan pelapor, dan penutupan laporan pengaduan melalui aplikasi SP4N-LAPOR.
""",
    },
    {
        "slug": "dasar-hukum",
        "title": "Dasar Hukum SOP Penyelesaian Pengaduan SP4N-LAPOR",
        "pages": "1, 3",
        "content": """
Dasar hukum SOP Penyelesaian Pengaduan SP4N-LAPOR meliputi:

1. Undang-Undang Nomor 25 Tahun 2009 tentang Pelayanan Publik.
2. Peraturan Menteri Pendayagunaan Aparatur Negara dan Reformasi Birokrasi Nomor 35 Tahun 2012 tentang Pedoman Penyusunan Standar Operasional Prosedur Administrasi Pemerintahan.
3. Peraturan Presiden Nomor 76 Tahun 2013 tentang Pengelolaan Pengaduan Pelayanan Publik.
4. Peraturan Presiden Nomor 95 Tahun 2018 tentang Sistem Pemerintahan Berbasis Elektronik (SPBE).
5. Peraturan Menteri Pendayagunaan Aparatur Negara dan Reformasi Birokrasi Nomor 62 Tahun 2018 tentang Pedoman Sistem Pengaduan Pelayanan Publik Nasional.
6. Peraturan Menteri Pendayagunaan Aparatur Negara dan Reformasi Birokrasi Nomor 46 Tahun 2020 tentang Roadmap Pengelolaan Pengaduan Pelayanan Publik Nasional.
7. Peraturan Menteri Dalam Negeri Nomor 8 Tahun 2023 tentang Pengelolaan Pengaduan di Lingkungan Kementerian Dalam Negeri dan Pemerintah Daerah.
8. Keputusan Menteri Pendayagunaan Aparatur Negara dan Reformasi Birokrasi Nomor 680 Tahun 2020 tentang Aplikasi Umum Bidang Pengelolaan Pengaduan Pelayanan Publik.
""",
    },
    {
        "slug": "kualifikasi-peralatan",
        "title": "Kualifikasi Pelaksana dan Peralatan SOP SP4N-LAPOR",
        "pages": "1",
        "content": """
Kualifikasi pelaksana SOP Penyelesaian Pengaduan SP4N-LAPOR:

1. Memiliki pemahaman pengelolaan pengaduan melalui aplikasi SP4N LAPOR!
2. Mampu mengoperasikan Komputer.
3. Memiliki pemahaman mengenai tugas pokok dan fungsi Perangkat Daerah.
4. Memiliki skill interpersonal.
5. Memiliki integritas terhadap kerahasiaan informasi.
6. Update terhadap perkembangan informasi.

Peralatan dan perlengkapan yang diperlukan:

1. Komputer.
2. Jaringan Internet.
""",
    },
    {
        "slug": "keterkaitan-peringatan-pencatatan",
        "title": "Keterkaitan Peringatan dan Pencatatan SOP SP4N-LAPOR",
        "pages": "1, 3",
        "content": """
Keterkaitan SOP:

1. Penyelesaian pengaduan melalui SP4N-LAPOR.
2. Pengaduan melalui aplikasi LAPOR.
3. Pengaduan LAPOR oleh Admin Koordinator dengan klasifikasi laporan pengaduan tidak berkadar pengawasan.
4. Pengaduan LAPOR oleh kanwil atau unit terkait dengan klasifikasi laporan pengaduan tidak berkadar pengawasan.

Peringatan:

1. Laporan yang tidak lengkap dan bukan kewenangan tidak dapat ditindaklanjuti.
2. Jika SOP ini tidak dijalankan, proses tindak lanjut pengaduan tidak dapat dilaksanakan.
3. Pelaksana bertanggung jawab atas pelaksanaan aktivitas yang telah dibakukan dan ditetapkan.

Pencatatan dan pendataan:

Semua pengaduan harus dicatat dan diarsipkan baik secara manual maupun elektronik.
""",
    },
    {
        "slug": "alur-verifikasi-admin",
        "title": "Alur Verifikasi Awal dan Klasifikasi Pengaduan SP4N-LAPOR",
        "pages": "2",
        "content": """
Pelaksana dalam alur verifikasi awal pengaduan SP4N-LAPOR terdiri atas tiga aktor: Admin Instansi, Inspektorat Jenderal, dan Pelapor.

Alur awal penyelesaian pengaduan melalui SP4N-LAPOR:

1. Admin Instansi melakukan login pada aplikasi LAPOR.
2. Admin Instansi mengelola laporan pengaduan masyarakat yang masuk melalui Sistem LAPOR dan kanal lain.
3. Admin Instansi menelaah substansi laporan untuk menentukan apakah laporan menjadi kewenangan satuan kerja, dengan memverifikasi unsur laporan pengaduan masyarakat.
4. Jika laporan bukan kewenangan Inspektorat Jenderal, laporan dikembalikan kepada Admin Instansi dan Admin Instansi menerima permintaan bukan wewenang.
5. Jika laporan menjadi kewenangan, Inspektorat Jenderal menerima laporan berkadar pengawasan dan mendisposisikannya.
6. Pelapor menerima notifikasi permintaan kelengkapan dokumen. Jika dokumen lengkap, pengaduan diteruskan; jika tidak, pengaduan tetap tertunda sampai dokumen dilengkapi.
7. Inspektorat Jenderal menindaklanjuti laporan dengan berkoordinasi dengan satuan kerja, mengikuti proses bisnis pada Inspektorat Jenderal.
8. Inspektorat Jenderal menyampaikan hasil telaah tindak lanjut laporan di sistem.
9. Pelapor menerima hasil tindak lanjut laporan.

Unsur laporan pengaduan masyarakat yang diverifikasi: identitas pelapor, tempat kejadian, waktu kejadian, kronologis kejadian, dan dokumen pendukung.
""",
    },
    {
        "slug": "mutu-baku-alur-verifikasi-admin",
        "title": "Mutu Baku Alur Verifikasi Pengaduan SP4N-LAPOR",
        "pages": "2",
        "content": """
Mutu baku setiap aktivitas pada alur verifikasi pengaduan SP4N-LAPOR, mencakup pelaksana, waktu, dan output:

1. Login pada Aplikasi LAPOR!, dilaksanakan Admin Instansi, memerlukan komputer/laptop dan akses internet, waktu 10 menit, output laporan pengaduan.
2. Mengelola laporan pengaduan masyarakat, dilaksanakan Admin Instansi, waktu 30 menit, output laporan yang sudah diperiksa.
3. Menelaah substansi laporan yang menjadi kewenangan satuan kerja, dilaksanakan Admin Instansi, waktu 30 menit, output laporan yang sudah diverifikasi.
4. Menerima permintaan bukan wewenang, dilaksanakan Admin Instansi, waktu 30 menit, output laporan yang sudah diverifikasi.
5. Menerima laporan berkadar pengawasan, dilaksanakan Inspektorat Jenderal, waktu 30 menit, output laporan yang sudah didisposisikan.
6. Tindak lanjut laporan minta kelengkapan dokumen pelapor, melibatkan Pelapor, waktu 30 menit, output laporan notifikasi minta kelengkapan dokumen pelapor.
7. Menindaklanjuti laporan dengan berkoordinasi dengan satuan kerja, dilaksanakan Inspektorat Jenderal, waktu 60 hari, output tindak lanjut laporan.
8. Menyampaikan hasil telaah tindak lanjut laporan di sistem, dilaksanakan Inspektorat Jenderal, waktu 30 menit, output telaah tindak lanjut laporan.
9. Menerima hasil tindak lanjut laporan, diterima Pelapor, waktu 30 menit, output hasil tindak lanjut laporan.

Total waktu keseluruhan alur ini adalah 60 hari.
""",
    },
    {
        "slug": "alur-perangkat-daerah",
        "title": "Alur Tindak Lanjut Pengaduan oleh Perangkat Daerah",
        "pages": "4",
        "content": """
Aktivitas tindak lanjut pengaduan oleh perangkat daerah:

1. Pejabat penghubung menerima disposisi laporan dari Admin Instansi melalui aplikasi.
2. Pejabat penghubung memberikan respon awal terhadap laporan melalui aplikasi.
3. Laporan dicetak dan diserahkan kepada pimpinan perangkat daerah untuk didisposisi.
4. Pimpinan perangkat daerah mendisposisikan laporan kepada pejabat pelaksana sesuai kewenangannya, seperti kepala bidang, kepala bagian, inspektur pembantu, atau jabatan fungsional yang disetarakan.
5. Pejabat pelaksana menjawab laporan secara lisan atau tertulis.
6. Laporan yang telah dijawab diserahkan kepada pejabat penghubung untuk diinput ke aplikasi SP4N-LAPOR.
7. Pejabat penghubung menginput jawaban dari pejabat berwenang ke aplikasi dan melampirkan data dukung jika ada.
8. Pejabat penghubung mengecek apakah laporan yang telah dijawab mendapat tanggapan dari pelapor melalui aplikasi.
9. Laporan ditutup melalui aplikasi setelah proses tindak lanjut selesai.

Pelaksana pada alur ini terdiri atas Pejabat Penghubung, Penanggung Jawab (Pimpinan Perangkat Daerah), dan Pejabat Pelaksana.
""",
    },
    {
        "slug": "mutu-baku-alur-perangkat-daerah",
        "title": "Mutu Baku Tindak Lanjut Pengaduan oleh Perangkat Daerah",
        "pages": "4",
        "content": """
Mutu baku setiap aktivitas tindak lanjut pengaduan oleh perangkat daerah, mencakup pelaksana, waktu, dan output:

1. Menerima disposisi laporan dari Admin Instansi melalui aplikasi, dilaksanakan Pejabat Penghubung, waktu 5 menit, output laporan yang telah diterima oleh Pejabat Penghubung.
2. Memberikan respon awal terhadap laporan melalui aplikasi, dilaksanakan Pejabat Penghubung, waktu 5 menit, output laporan yang telah direspon awal.
3. Mencetak laporan dan menyerahkan kepada Pimpinan Perangkat Daerah untuk didisposisi, dilaksanakan Pejabat Penghubung, waktu maksimal 3 hari, output laporan yang dicetak dan diserahkan.
4. Mendisposisi laporan ke Pejabat Pelaksana sesuai kewenangannya, dilaksanakan Penanggung Jawab atau Pimpinan Perangkat Daerah, waktu 5 menit, output laporan yang terdisposisi ke pejabat Eselon III atau IV yang berwenang.
5. Menjawab laporan secara lisan atau tertulis, dilaksanakan Pejabat Pelaksana, output laporan yang telah mendapat jawaban.
6. Menyerahkan laporan yang telah dijawab kepada Pejabat Penghubung untuk diinput ke aplikasi SP4N-LAPOR, dilaksanakan Pejabat Pelaksana, waktu 10 menit, output laporan yang telah diserahkan.
7. Menginput jawaban dari pejabat berwenang ke dalam aplikasi dan melampirkan data dukung jika ada, dilaksanakan Pejabat Penghubung, waktu 10 menit, output jawaban laporan diinput di aplikasi.
8. Melakukan pengecekan apakah laporan yang telah dijawab mendapatkan tanggapan dari pelapor melalui aplikasi, dilaksanakan Pejabat Penghubung, output laporan yang telah dicek.
9. Menutup laporan melalui aplikasi, dilaksanakan Pejabat Penghubung, waktu 5 menit, output laporan selesai.
""",
    },
    {
        "slug": "waktu-penyelesaian",
        "title": "Waktu Penyelesaian Pengaduan dalam SOP SP4N-LAPOR",
        "pages": "2, 4",
        "content": """
SOP mencantumkan waktu mutu baku dalam proses penyelesaian pengaduan:

1. Login pada aplikasi LAPOR dicatat dengan waktu 10 menit.
2. Sebagian besar aktivitas verifikasi dan penelaahan laporan oleh Admin Instansi maupun Inspektorat Jenderal dicatat dengan waktu 30 menit per aktivitas.
3. Aktivitas administratif pada perangkat daerah, seperti menerima disposisi, memberikan respon awal, mendisposisi laporan, dan menutup laporan, dicatat dengan waktu 5 menit.
4. Menginput jawaban ke aplikasi dan menyerahkan laporan yang telah dijawab dicatat dengan waktu 10 menit.
5. Mencetak laporan dan menyerahkannya kepada Pimpinan Perangkat Daerah untuk didisposisi memerlukan waktu maksimal 3 hari.
6. Permintaan informasi dan pengaduan yang bersifat normatif ditangani dalam 5 hari kerja.
7. Laporan tidak berkadar pengawasan ditangani dalam 14 hari kerja.
8. Laporan berkadar pengawasan ditangani dalam 60 hari kerja.
9. Menindaklanjuti laporan dengan berkoordinasi dengan satuan kerja dicatat dengan waktu 60 hari.
10. Setelah jawaban terakhir diberikan, laporan dicek setiap hari selama 10 hari kerja untuk melihat tanggapan pelapor melalui aplikasi.

Total waktu pada alur verifikasi dan tindak lanjut pengaduan SP4N-LAPOR adalah 60 hari.
""",
    },
    {
        "slug": "identitas-dasar-hukum-lapor",
        "title": "Identitas dan Dasar Hukum SOP Pengelolaan Pengaduan LAPOR",
        "pages": "3",
        "dokumen": "SOP Pengelolaan Pengaduan LAPOR",
        "content": """
SOP ini mengatur pengelolaan pengaduan melalui aplikasi LAPOR di lingkungan Pemerintah Kabupaten Bandung -- dokumen ini terpisah dari SOP Penyelesaian Pengaduan SP4N-LAPOR di atas.

Nama SOP: Pengelolaan Pengaduan LAPOR!.

SOP disahkan oleh Kepala Dinas Komunikasi dan Informatika, Statistik dan Persandian Kabupaten Bandung.

Dasar hukum SOP Pengelolaan Pengaduan LAPOR! meliputi:

1. Undang-Undang Nomor 25 Tahun 2009 tentang Pelayanan Publik.
2. Peraturan Menteri Pendayagunaan Aparatur Negara dan Reformasi Birokrasi Nomor 35 Tahun 2012 tentang Pedoman Penyusunan Standar Operasional Prosedur Administrasi Pemerintahan.
3. Peraturan Presiden Nomor 76 Tahun 2013 tentang Pengelolaan Pengaduan Pelayanan Publik.
4. Peraturan Presiden Nomor 95 Tahun 2018 tentang Sistem Pemerintahan Berbasis Elektronik (SPBE).
5. Peraturan Menteri Pendayagunaan Aparatur Negara dan Reformasi Birokrasi Nomor 62 Tahun 2018 tentang Pedoman Sistem Pengaduan Pelayanan Publik Nasional.
6. Peraturan Menteri Pendayagunaan Aparatur Negara dan Reformasi Birokrasi Nomor 46 Tahun 2020 tentang Roadmap Pengelolaan Pengaduan Pelayanan Publik Nasional.
7. Peraturan Menteri Dalam Negeri Nomor 8 Tahun 2023 tentang Pengelolaan Pengaduan di Lingkungan Kementerian Dalam Negeri dan Pemerintah Daerah.
8. Keputusan Menteri Pendayagunaan Aparatur Negara dan Reformasi Birokrasi Nomor 680 Tahun 2020 tentang Aplikasi Umum Bidang Pengelolaan Pengaduan Pelayanan Publik.
""",
    },
    {
        "slug": "kualifikasi-alur-peralatan-lapor",
        "title": "Kualifikasi Pelaksana, Alur, dan Peralatan SOP Pengelolaan Pengaduan LAPOR",
        "pages": "3",
        "dokumen": "SOP Pengelolaan Pengaduan LAPOR",
        "content": """
Kualifikasi pelaksana SOP Pengelolaan Pengaduan LAPOR!:

1. Memahami peraturan pengelolaan pengaduan masyarakat melalui aplikasi.
2. Memahami mekanisme pengelolaan pengaduan masyarakat melalui aplikasi. [Catatan: kalimat ini terpotong di dokumen sumber, tidak ada lanjutannya pada kolom tabel PDF asli.]
3. Memiliki kemampuan mengelola pengaduan masyarakat melalui aplikasi.
4. Memiliki kemampuan mengkoordinasikan, mengklarifikasi dan menindaklanjuti laporan. [Catatan: kalimat ini terpotong di dokumen sumber, tidak ada lanjutannya pada kolom tabel PDF asli -- bagian setelah "dan" adalah perkiraan konteks, PERLU VERIFIKASI ke dokumen asli.]
5. Memiliki kemampuan menyunting redaksional atau penggunaan bahasa.

Alur proses pengaduan pada SOP ini mencakup:

1. Pengaduan melalui aplikasi LAPOR!.
2. Kategorisasi pengaduan LAPOR! oleh Admin Koordinator dengan klasifikasi laporan pengaduan tidak berkadar pengawasan.
3. Kategorisasi pengaduan LAPOR! oleh Kanwil dengan klasifikasi laporan pengaduan tidak berkadar pengawasan.

Catatan: SOP ini di dalam dokumen PDF hanya memuat identitas, dasar hukum, kualifikasi pelaksana, alur proses ringkas di atas, peralatan, dan pencatatan -- TIDAK memuat tabel aktivitas/pelaksana/mutu baku terperinci seperti pada SOP Penyelesaian Pengaduan SP4N-LAPOR (kemungkinan halaman lanjutannya tidak termasuk dalam PDF yang tersedia).

Peralatan dan perlengkapan yang diperlukan:

1. PC.
2. Akses internet.
3. Form Disposisi.
4. Form Jawaban.
5. Perangkat pendukung: printer, scanner.

Pencatatan dan pendataan:

Semua pengaduan harus dicatat dan diarsipkan baik secara manual maupun elektronik.
""",
    },
]


def chunk_text(text):
    if len(text) <= CHUNK_SIZE:
        return [text]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=False,
    )
    return [c.strip() for c in splitter.split_text(text) if c.strip()]


def sanitize_filename(value):
    value = re.sub(r'[\\/*?:"<>|]', "_", value)
    value = re.sub(r"\s+", "_", value)
    return value.strip("_")


def slugify(value):
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "sop"


def yaml_quote(value):
    value = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{value}"'


def yaml_string_list(values):
    quoted = ", ".join(yaml_quote(value) for value in values)
    return f"[{quoted}]"


def parse_pages(raw_pages):
    return [page.strip() for page in str(raw_pages).split(",") if page.strip()]


def build_markdown(section, content, sub_idx=None):
    title = section["title"]
    pages = parse_pages(section["pages"])
    doc_name = section.get("dokumen", DOC_NAME)
    source_id = slugify(doc_name)
    chunk_id = (
        f"{source_id}-{section['slug']}-{sub_idx:02d}"
        if sub_idx is not None
        else f"{source_id}-{section['slug']}"
    )

    return f"""---
source_id: {yaml_quote(source_id)}
chunk_id: {yaml_quote(chunk_id)}
jenis: {yaml_quote("regulasi")}
tipe_dokumen: {yaml_quote("sop")}
dokumen: {yaml_quote(doc_name)}
sumber_file: {yaml_quote(SOURCE_FILE)}
halaman: {yaml_string_list(pages)}
section: {yaml_quote(title)}
---

# {title}

{content}
"""


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"File SOP tidak ditemukan: {PDF_PATH}")

    print(f"[LOAD] {PDF_PATH}")
    print(f"[BUILD] Generate corpus SOP to: {OUTPUT_DIR}")

    for stale in OUTPUT_DIR.glob("SOP_Pengaduan_*.md"):
        stale.unlink()
        print(f"[CLEAR] {stale.name}")

    for index, section in enumerate(SECTIONS):
        filename = sanitize_filename(
            f"SOP_Pengaduan_{index:02d}_{section['slug']}"
        ) + ".md"
        filepath = OUTPUT_DIR / filename
        filepath.write_text(
            build_markdown(section, section["content"].strip()), encoding="utf-8"
        )
        print(f"[SAVE] {filepath.name}")

    print(f"[DONE] Total corpus SOP: {len(SECTIONS)} seksi")


if __name__ == "__main__":
    main()