import json
import os
from pathlib import Path
import re

import chromadb
import torch
import yaml
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
EMBED_MODEL_SLUG = os.getenv("EMBED_MODEL_SLUG", "intfloat--multilingual-e5-base")
MODEL_GLOB = f"models--{EMBED_MODEL_SLUG}/snapshots/*"
REQUIRED_CORPUS_DIRS = [
    BASE_DIR / "corpus_aduan",
    BASE_DIR / "corpus_regulasi_sop",
]
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(BASE_DIR / "chroma_rag")))
COLLECTION_NAME = "pengaduan_regulasi"
BATCH_SIZE = 64


def find_model_path():
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    matches = sorted(cache_dir.glob(MODEL_GLOB))

    if not matches:
        raise FileNotFoundError(
            "Model embedding lokal tidak ditemukan di cache Hugging Face."
        )

    return matches[-1]


def get_embedding_device():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA tidak tersedia. Embedding model dikonfigurasi untuk GPU saja."
        )

    return "cuda"


def format_passage_for_embedding(text):
    return f"passage: {str(text).strip()}"


def split_frontmatter(raw_text):
    if not raw_text.startswith("---\n"):
        return {}, raw_text

    parts = raw_text.split("---\n", 2)
    if len(parts) < 3:
        return {}, raw_text

    metadata = yaml.safe_load(parts[1]) or {}
    content = parts[2]
    return metadata, content


def load_documents():
    documents = []
    corpus_dirs = REQUIRED_CORPUS_DIRS

    for corpus_dir in corpus_dirs:
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


def chunked(items, size):
    for index in range(0, len(items), size):
        yield items[index:index + size]


def main():
    print("[LOAD] Read markdown corpus...")
    for required_dir in REQUIRED_CORPUS_DIRS:
        if not required_dir.exists():
            raise FileNotFoundError(
                f"Folder korpus wajib tidak ditemukan: {required_dir}"
            )

    print("[INFO] Corpus sources:")
    for corpus_dir in REQUIRED_CORPUS_DIRS:
        print(f" - wajib   : {corpus_dir}")

    documents = load_documents()

    if not documents:
        raise RuntimeError("Tidak ada dokumen markdown untuk diindeks.")

    print(f"[INFO] Total chunks: {len(documents)}")

    model_path = find_model_path()
    print(f"[LOAD] Embedding model: {model_path}")
    print(f"[LOAD] Embedding device: {get_embedding_device()}")
    model = SentenceTransformer(
        str(model_path),
        device=get_embedding_device(),
    )

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "hnsw:construction_ef": 100,
            "hnsw:search_ef": 100,
        },
    )

    print("[BUILD] Encode and upsert to ChromaDB...")

    for batch in chunked(documents, BATCH_SIZE):
        ids = [item["id"] for item in batch]
        metadatas = [item["metadata"] for item in batch]
        texts = [build_search_text(item) for item in batch]
        embeddings = model.encode(
            [format_passage_for_embedding(text) for text in texts],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    print(f"[DONE] ChromaDB saved to: {CHROMA_DIR}")
    print(f"[DONE] Collection: {COLLECTION_NAME}")


if __name__ == "__main__":
    main()
