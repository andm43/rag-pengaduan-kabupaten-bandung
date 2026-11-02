import json
import math
import pickle
import re
from collections import Counter
from pathlib import Path

import yaml


BASE_DIR = Path(__file__).resolve().parent
REQUIRED_CORPUS_DIRS = [
    BASE_DIR / "corpus_aduan",
    BASE_DIR / "corpus_regulasi_sop",
]
OUTPUT_PATH = BASE_DIR / "chroma_rag" / "bm25_index.pkl"

BM25_K1 = 1.5
BM25_B = 0.75


def split_frontmatter(raw_text):
    if not raw_text.startswith("---\n"):
        return {}, raw_text

    parts = raw_text.split("---\n", 2)
    if len(parts) < 3:
        return {}, raw_text

    metadata = yaml.safe_load(parts[1]) or {}
    content = parts[2]
    return metadata, content


def normalize_metadata(metadata):
    normalized = {}

    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            normalized[key] = value
        elif value is None:
            continue
        else:
            normalized[key] = json.dumps(value, ensure_ascii=False)

    return normalized


def load_documents():
    documents = []

    for corpus_dir in REQUIRED_CORPUS_DIRS:
        for path in sorted(corpus_dir.glob("*.md")):
            raw_text = path.read_text(encoding="utf-8")
            metadata, content = split_frontmatter(raw_text)

            metadata["path"] = str(path)
            documents.append(
                {
                    "id": str(metadata.get("chunk_id", path.stem)),
                    "document": content.strip(),
                    "metadata": normalize_metadata(metadata),
                }
            )

    return documents


def build_search_text(item):
    metadata = item["metadata"]
    document = re.sub(r"\[[A-Z\s]+\]", " ", item["document"])
    document = re.sub(r"\s+", " ", document).strip()

    parts = [
        metadata.get("tipe_dokumen", ""),
        metadata.get("dokumen", ""),
        metadata.get("section", ""),
        metadata.get("judul", ""),
        metadata.get("kategori", ""),
        metadata.get("kecamatan", ""),
        metadata.get("instansi", ""),
        document,
    ]

    return "\n".join(str(part).strip() for part in parts if str(part).strip())


def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text):
    return normalize_text(text).split()


def main():
    print("[LOAD] Baca korpus markdown...")
    for required_dir in REQUIRED_CORPUS_DIRS:
        if not required_dir.exists():
            raise FileNotFoundError(f"Folder korpus wajib tidak ditemukan: {required_dir}")

    documents = load_documents()
    if not documents:
        raise RuntimeError("Tidak ada dokumen markdown untuk diindeks.")

    print(f"[INFO] Total chunk: {len(documents)}")

    print("[TOKENIZE] Tokenisasi seluruh chunk...")
    chunk_ids = []
    metadatas = []
    tokenized_docs = []

    for item in documents:
        search_text = build_search_text(item)
        chunk_ids.append(item["id"])
        metadatas.append(
            {
                "source_id": item["metadata"].get("source_id"),
                "jenis": item["metadata"].get("jenis"),
                "tipe_dokumen": item["metadata"].get("tipe_dokumen"),
            }
        )
        tokenized_docs.append(tokenize(search_text))

    print("[INDEX] Hitung IDF & panjang dokumen (BM25Okapi)...")
    N = len(tokenized_docs)
    doc_lens = [len(d) for d in tokenized_docs]
    avgdl = sum(doc_lens) / N if N else 0.0

    df = Counter()
    for d in tokenized_docs:
        for term in set(d):
            df[term] += 1

    idf = {
        term: math.log((N - freq + 0.5) / (freq + 0.5) + 1)
        for term, freq in df.items()
    }

    term_freqs = [Counter(d) for d in tokenized_docs]

    print(f"[INFO] Vocab size: {len(df)}")

    index = {
        "chunk_ids": chunk_ids,
        "metadatas": metadatas,
        "term_freqs": term_freqs,
        "doc_lens": doc_lens,
        "avgdl": avgdl,
        "idf": idf,
        "N": N,
        "k1": BM25_K1,
        "b": BM25_B,
    }

    with OUTPUT_PATH.open("wb") as f:
        pickle.dump(index, f)

    print(f"[DONE] BM25 index disimpan ke: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
