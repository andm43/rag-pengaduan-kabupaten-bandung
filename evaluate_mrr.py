import argparse
import json
from pathlib import Path
from statistics import mean

from ask_rag import retrieve_context, set_chroma_dir, normalize_query_text


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = BASE_DIR / "golden_dataset_fix.jsonl"


def load_eval_samples(path):
    samples = []

    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))

    return samples


def is_answerable(sample):
    has_chunk_targets = bool(sample.get("expected_chunk_ids"))
    has_source_targets = bool(sample.get("expected_source_ids"))
    return has_chunk_targets or has_source_targets


def resolve_retrieval_filters(sample, use_source_scope=True):
    return {
        "jenis": sample.get("jenis_filter") or sample.get("jenis"),
        "tipe_dokumen": sample.get("tipe_dokumen_filter") or sample.get("tipe_dokumen"),
        "source_id": sample.get("source_id_filter") if use_source_scope else None,
    }


def match_rank(items, sample):
    expected_chunk_ids = set(sample.get("expected_chunk_ids") or [])
    expected_source_ids = set(sample.get("expected_source_ids") or [])

    for rank, item in enumerate(items, start=1):
        metadata = item["metadata"]
        chunk_id = metadata.get("chunk_id", item["chunk_id"])
        source_id = metadata.get("source_id")

        if chunk_id in expected_chunk_ids:
            return rank, "chunk_id", chunk_id, source_id
        if source_id in expected_source_ids:
            return rank, "source_id", chunk_id, source_id

    return None, None, None, None


def evaluate_sample(sample, top_k, use_source_scope=True):
    retrieval_filters = resolve_retrieval_filters(sample, use_source_scope=use_source_scope)
    items = retrieve_context(
        query=normalize_query_text(sample["question"]),
        top_k=top_k,
        jenis=retrieval_filters["jenis"],
        tipe_dokumen=retrieval_filters["tipe_dokumen"],
        source_id=retrieval_filters["source_id"],
    )
    rank, matched_by, matched_chunk_id, matched_source_id = match_rank(items, sample)
    reciprocal_rank = 0.0 if rank is None else 1.0 / rank

    return {
        "id": sample["id"],
        "question": sample["question"],
        "behavior": sample.get("behavior"),
        "answerable": is_answerable(sample),
        "source_scoped": use_source_scope and bool(retrieval_filters["source_id"]),
        "rank": rank,
        "matched_by": matched_by,
        "matched_chunk_id": matched_chunk_id,
        "matched_source_id": matched_source_id,
        "reciprocal_rank": reciprocal_rank,
        "top_hit_chunk_id": (
            items[0]["metadata"].get("chunk_id", items[0]["chunk_id"])
            if items else None
        ),
        "top_hit_source_id": (
            items[0]["metadata"].get("source_id")
            if items else None
        ),
        "top_hit_score": (items[0]["score"] if items else None),
        "top_hit_base_score": (items[0]["base_score"] if items else None),
        "top_hit_rerank_score": (items[0]["rerank_score"] if items else None),
        "expected_source_ids": sample.get("expected_source_ids") or [],
        "expected_chunk_ids": sample.get("expected_chunk_ids") or [],
        "retrieval_filters": retrieval_filters,
    }


def summarize(results):
    answerable = [r for r in results if r["answerable"]]
    unanswerable = [r for r in results if not r["answerable"]]

    mrr_answerable = mean(r["reciprocal_rank"] for r in answerable) if answerable else 0.0
    mrr_naive = mean(r["reciprocal_rank"] for r in results) if results else 0.0

    return {
        "mrr": mrr_answerable,
        "mrr_naive": mrr_naive,
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        "unanswerable_results": unanswerable,
    }


def write_results(output_path, payload):
    target_path = Path(output_path)
    content = json.dumps(payload, ensure_ascii=False, indent=2)

    try:
        target_path.write_text(content, encoding="utf-8")
        return target_path, None
    except PermissionError:
        fallback_path = target_path.with_name(f"{target_path.stem}_latest{target_path.suffix}")
        fallback_path.write_text(content, encoding="utf-8")
        warning = (
            f"File output utama terkunci: {target_path}. "
            f"Hasil disimpan ke: {fallback_path}."
        )
        return fallback_path, warning


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Path ke file JSONL golden dataset evaluasi.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--output",
        default=str(BASE_DIR / "mrr_results.json"),
        help="Path file JSON hasil evaluasi MRR.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Opsional. Batasi jumlah sampel yang dievaluasi dari file dataset.",
    )
    parser.add_argument(
        "--chroma-dir",
        default=None,
        help="Opsional. Path ke direktori ChromaDB. Default: chroma_rag/ di samping script.",
    )
    parser.add_argument(
        "--no-source-scope",
        action="store_true",
        help=(
            "Matikan pembatasan retrieval ke source_id_filter (retrieval "
            "dilakukan ke SELURUH korpus, bukan hanya ke 1 dokumen sumber "
            "yang sudah diketahui). Gunakan ini untuk mengukur kinerja "
            "retrieval kondisi produksi (sistem belum tahu dokumen mana "
            "yang relevan)."
        ),
    )
    args = parser.parse_args()

    if args.chroma_dir:
        set_chroma_dir(args.chroma_dir)

    samples = load_eval_samples(args.dataset)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    use_source_scope = not args.no_source_scope
    results = [
        evaluate_sample(sample, top_k=args.top_k, use_source_scope=use_source_scope)
        for sample in samples
    ]
    summary = summarize(results)

    payload = {
        "dataset": str(Path(args.dataset).resolve()),
        "top_k": args.top_k,
        "source_scoped_retrieval": use_source_scope,
        "sample_count": len(results),
        **summary,
        "results": results,
    }

    saved_path, save_warning = write_results(args.output, payload)

    print(f"[DONE] Mode retrieval : {'dibatasi ke source_id_filter' if use_source_scope else 'full-corpus'}")
    print(f"[DONE] MRR            : {summary['mrr']:.4f}  (n={summary['answerable_count']}, {summary['unanswerable_count']} sampel not_found dikecualikan)")
    print(f"[DONE] MRR (semua)    : {summary['mrr_naive']:.4f}  (n={len(results)}, hanya pembanding)")
    print(f"[DONE] Hasil detail   : {saved_path}")
    if save_warning:
        print(f"- save_warning: {save_warning}")

    print()
    for item in results:
        rank_text = item["rank"] if item["rank"] is not None else "-"
        note = "" if item["answerable"] else "  (not_found, di luar MRR)"
        print(
            f"- {item['id']}: RR={item['reciprocal_rank']:.4f} | "
            f"rank={rank_text} | top={item['top_hit_chunk_id']}{note}"
        )


if __name__ == "__main__":
    main()