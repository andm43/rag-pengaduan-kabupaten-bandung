import argparse
import asyncio
import json
import math
import os
import re
import sys
import time
import warnings
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"Importing .* from 'ragas\.metrics' is deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"LangchainLLMWrapper is deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"LangchainEmbeddingsWrapper is deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"ContextPrecision is deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"ContextRecall is deprecated.*",
)

from ragas import EvaluationDataset, evaluate
from ragas.cost import get_token_usage_for_openai, TokenUsage
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness, ResponseRelevancy, SemanticSimilarity
from ragas.run_config import RunConfig

import ask_rag as _ask_rag
from ask_rag import (
    answer_query,
    find_model_path,
    generate_answer,
    get_embedding_device,
    set_ollama_url,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = BASE_DIR / "golden_dataset_fix.jsonl"
METRIC_NAMES = ["faithfulness", "answer_relevancy", "semantic_similarity"]
CONTEXT_METRIC_NAMES = ["context_precision", "context_recall"]
NO_RAG_METRIC_NAMES = ["answer_relevancy", "semantic_similarity"]
NO_RAG_SYSTEM_PROMPT = (
    "Anda adalah asisten yang menjawab pertanyaan seputar pengaduan masyarakat "
    "dan dokumen regulasi di Kabupaten Bandung. Jawab pertanyaan pengguna sebaik "
    "mungkin berdasarkan pengetahuan Anda sendiri. Jawab ringkas dalam Bahasa Indonesia."
)

JUDGE_SEED = 42

FAITHFULNESS_SCOPE_EXCLUDED_MODES = {
    "disposition",
    "beneficiary_distribution_unsupported",
    "operator_handoff",
    "out_of_scope_region",
    "out_of_scope_service",
    "insufficient_context",
    "need_clarification",
    "regulation_clarify",
    "pasal_not_found",
    "tracking_not_found",
    "identity_lookup_declined",
    "new_complaint_reset",
    "topic_drift_repair",
    "acknowledgement",
    "service_info_handoff",
}

OPENAI_PRICING = {
    "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
    "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
}

ANTHROPIC_PRICING = {
    "claude-sonnet-4-6": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
    "claude-haiku-4-5-20251001": {"input": 1.00 / 1_000_000, "output": 5.00 / 1_000_000},
}


def get_token_usage_for_anthropic(llm_result):

    input_tokens = 0
    output_tokens = 0
    for generation_list in llm_result.generations:
        for generation in generation_list:
            message = getattr(generation, "message", None)
            usage = getattr(message, "usage_metadata", None) if message else None
            if usage:
                input_tokens += usage.get("input_tokens", 0) or 0
                output_tokens += usage.get("output_tokens", 0) or 0
    return TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens, model="")


if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def load_eval_samples(path):
    samples = []

    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))

    return samples


def resolve_retrieval_filters(sample, use_source_scope=True):
    return {
        "jenis": sample.get("jenis_filter") or sample.get("jenis"),
        "tipe_dokumen": sample.get("tipe_dokumen_filter") or sample.get("tipe_dokumen"),
        "source_id": sample.get("source_id_filter") if use_source_scope else None,
    }


def is_out_of_faithfulness_scope(row):
    """True kalau sampel tidak layak dinilai faithfulness/answer_relevancy.

    Dua kriteria:
    1. mode termasuk FAITHFULNESS_SCOPE_EXCLUDED_MODES (jawaban template /
       penolakan guardrail, bukan hasil generasi dari konteks), atau
    2. tidak ada konteks sama sekali -- faithfulness secara definisi tidak
       terdefinisi tanpa konteks, alasan yang sama dipakai --no-rag.

    Kriteria 2 dipakai sebagai pengaman berbasis data, supaya tidak perlu
    mengejar setiap nama mode guardrail baru satu per satu di kemudian hari.
    """
    return (
        row["mode"] in FAITHFULNESS_SCOPE_EXCLUDED_MODES
        or len(row.get("context_items") or []) == 0
    )


def format_context_for_ragas(item):
    """Rakit satu konteks untuk RAGAS lengkap dengan metadata.

    Field yang disertakan mengikuti build_context_block() di ask_rag.py (status,
    tanggal_masuk, durasi_respon, dll.), tanpa base_score/rerank_score, sehingga
    juri menilai jawaban dengan konteks yang sama seperti yang dilihat generator.
    Ditulis terpisah agar jalur produksi dan perhitungan MRR tidak tersentuh.
    """
    metadata = item.get("metadata") or {}
    parts = [
        f"chunk_id: {metadata.get('chunk_id', item.get('chunk_id', '-'))}",
        f"source_id: {metadata.get('source_id', '-')}",
        f"jenis: {metadata.get('jenis', '-')}",
        f"tipe_dokumen: {metadata.get('tipe_dokumen', '-')}",
        f"dokumen: {metadata.get('dokumen', '-')}",
        f"section: {metadata.get('section', '-')}",
    ]

    for key, value in (
        ("status", metadata.get("status")),
        ("tanggal_masuk", metadata.get("tanggal_masuk")),
        ("tanggal_tindak_lanjut_pertama", metadata.get("tanggal_tindak_lanjut_pertama")),
        ("tanggal_selesai", metadata.get("tanggal_selesai")),
        ("durasi_respon_hari", metadata.get("durasi_respon")),
        ("durasi_penyelesaian_hari", metadata.get("durasi_penyelesaian")),
    ):
        if value not in (None, "", "-"):
            parts.append(f"{key}: {value}")

    parts += ["isi:", item.get("document", "")]
    return "\n".join(parts)


def resolve_reference_answer(sample):
    return (
        sample.get("reference_answer_short")
        or sample.get("reference_answer")
        or sample.get("ground_truth")
        or ""
    )


def resolve_reference_answer_full(sample):
    return (
        sample.get("reference_answer")
        or sample.get("ground_truth")
        or sample.get("reference_answer_short")
        or ""
    )


def build_prediction_rows(samples, model_name, top_k, temperature, no_rag=False, use_source_scope=True):
    rows = []
    total = len(samples)
    mode_tally = {}

    for index, sample in enumerate(samples, 1):
        retrieval_filters = resolve_retrieval_filters(sample, use_source_scope=use_source_scope)
        if no_rag:
            answer_text = generate_answer(
                model_name=model_name,
                prompt=sample["question"],
                temperature=temperature,
                system_prompt_override=NO_RAG_SYSTEM_PROMPT,
            )
            result = {
                "answer": answer_text,
                "mode": "no_rag",
                "used_generation": True,
                "context_items": [],
            }
            retrieval_filters = {"jenis": None, "tipe_dokumen": None, "source_id": None}
        else:
            result = answer_query(
                query=sample["question"],
                model_name=model_name,
                top_k=top_k,
                jenis=retrieval_filters["jenis"],
                tipe_dokumen=retrieval_filters["tipe_dokumen"],
                source_id=retrieval_filters["source_id"],
                temperature=temperature,
                chat_history=None,
                allow_chat_fallback=False,
            )
        mode_tally[result["mode"]] = mode_tally.get(result["mode"], 0) + 1
        print(f"\r  Generate jawaban: {index}/{total}", end="", flush=True)
        rows.append(
            {
                "id": sample["id"],
                "question": sample["question"],
                "mode": result["mode"],
                "used_generation": result.get("used_generation", False),
                "raw_answer": result["answer"],
                "evaluated_answer": strip_direct_answer_label(result["answer"]),
                "answer": result["answer"],
                "contexts": [
                    format_context_for_ragas(item) for item in result["context_items"]
                ],
                "contexts_raw": [item["document"] for item in result["context_items"]],
                "context_items": result["context_items"],
                "reference_answer": resolve_reference_answer(sample),
                "reference_answer_full": resolve_reference_answer_full(sample),
                "source_count": len(result["context_items"]),
                "expected_source_ids": sample.get("expected_source_ids") or [],
                "expected_chunk_ids": sample.get("expected_chunk_ids") or [],
                "retrieval_filters": retrieval_filters,
            }
        )

    if total:
        mode_summary = ", ".join(f"{k}={v}" for k, v in sorted(mode_tally.items()))
        print(f"\r  Generate jawaban: {total}/{total} selesai  ({mode_summary})")

    return rows


def clean_metric_value(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def mean_ignore_none(values):
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return float(sum(filtered) / len(filtered))


def strip_direct_answer_label(text):
    stripped = str(text).strip()
    patterns = [
        r"^jawaban instansi\s*:\s*",
        r"^respons instansi\s*:\s*",
        r"^respon instansi\s*:\s*",
        r"^tindak lanjut instansi\s*:\s*",
    ]
    for pattern in patterns:
        stripped = re.sub(pattern, "", stripped, flags=re.IGNORECASE)
    return stripped.strip()


def to_ragas_dataset(rows):
    samples = []

    for row in rows:
        samples.append(
            {
                "user_input": row["question"],
                "response": row["evaluated_answer"],
                "retrieved_contexts": row["contexts"],
                "reference": row["reference_answer"],
            }
        )

    return EvaluationDataset.from_list(samples)


def build_ragas_metrics(llm_model, include_context_metrics=False):
    embedding_model = HuggingFaceEmbeddings(
        model_name=str(find_model_path()),
        model_kwargs={"device": get_embedding_device()},
        encode_kwargs={"normalize_embeddings": True},
    )

    ragas_llm = LangchainLLMWrapper(llm_model)
    ragas_embeddings = LangchainEmbeddingsWrapper(embedding_model)

    metrics = {
        "faithfulness": Faithfulness(llm=ragas_llm, max_retries=2),
        "answer_relevancy": ResponseRelevancy(
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            strictness=3,
        ),
        "semantic_similarity": SemanticSimilarity(embeddings=ragas_embeddings),
    }

    if include_context_metrics:
        metrics["context_precision"] = ContextPrecision(llm=ragas_llm, max_retries=2)
        metrics["context_recall"] = ContextRecall(llm=ragas_llm, max_retries=2)

    return metrics


def evaluate_metric(dataset, metric_name, metric, run_config, token_usage_parser=None):
    try:
        result = evaluate(
            dataset=dataset,
            metrics=[metric],
            run_config=run_config,
            show_progress=True,
            allow_nest_asyncio=False,
            **({"token_usage_parser": token_usage_parser} if token_usage_parser else {}),
        )
        scores_df = result.to_pandas()
        token_usage = None
        if token_usage_parser:
            try:
                token_usage = result.total_tokens()
            except Exception:
                token_usage = None
        if metric_name not in scores_df.columns:
            return [None] * len(dataset), f"Kolom metric '{metric_name}' tidak ditemukan.", token_usage
        return [
            clean_metric_value(value)
            for value in scores_df[metric_name].tolist()
        ], None, token_usage
    except Exception as exc:
        return [None] * len(dataset), str(exc), None


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
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument(
        "--judge-provider",
        default="ollama",
        choices=["ollama", "anthropic", "deepseek", "openai"],
        help="Provider LLM-as-a-Judge untuk RAGAS. 'ollama' (default), 'anthropic' (Claude), 'openai' (GPT), atau 'deepseek' (DeepSeek).",
    )
    parser.add_argument(
        "--evaluator-model",
        default=None,
        help="Model evaluator untuk RAGAS. Default: ikuti --model (ollama), claude-haiku-4-5-20251001 (anthropic), gpt-4o-mini (openai), atau deepseek-chat (deepseek).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help=(
            "Jumlah request LLM judge yang diproses paralel (default 2, "
            "diturunkan dari 4 setelah run 100 soal sempat kena beberapa "
            "APIConnectionError). Turunkan ke 1 kalau masih sering putus "
            "koneksi; naikkan ke 4 kalau koneksi stabil dan mau lebih cepat."
        ),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--ollama-url",
        default=None,
        help="Endpoint Ollama /api/generate. Gunakan ini saat evaluasi diarahkan ke VPS.",
    )
    parser.add_argument(
        "--output",
        default=str(BASE_DIR / "ragas_results.json"),
        help="Path file JSON hasil evaluasi RAGAS.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Opsional. Batasi jumlah sampel yang dievaluasi dari file dataset.",
    )
    parser.add_argument(
        "--context-metrics",
        action="store_true",
        default=False,
        help="Tambahkan evaluasi Context Precision dan Context Recall (membutuhkan LLM judge, menambah waktu dan biaya).",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        default=False,
        help=(
            "Baseline closed-book: lewati retrieval sepenuhnya, model hanya diberi "
            "pertanyaan tanpa konteks. Hanya answer_relevancy dan semantic_similarity "
            "yang dihitung -- faithfulness dan context metrics tidak terdefinisi tanpa konteks."
        ),
    )
    parser.add_argument(
        "--no-source-scope",
        action="store_true",
        default=False,
        help=(
            "Matikan pembatasan retrieval ke source_id_filter (retrieval dilakukan "
            "ke SELURUH korpus, bukan hanya ke 1 dokumen sumber yang sudah diketahui). "
            "Gunakan ini untuk mengukur kinerja RAG kondisi produksi (sistem belum tahu "
            "dokumen mana yang relevan). Setara dengan --no-source-scope di evaluate_mrr.py. "
            "Dataset yang dipakai tetap file --dataset apa adanya (mis. golden_dataset_fix.jsonl) "
            "-- flag ini yang mengabaikan source_id_filter di dalamnya saat runtime, bukan file."
        ),
    )
    args = parser.parse_args()

    if args.no_rag and args.context_metrics:
        print("[WARN] --context-metrics diabaikan karena --no-rag tidak menghasilkan konteks.")
        args.context_metrics = False

    if args.ollama_url:
        set_ollama_url(args.ollama_url)

    eval_samples = load_eval_samples(args.dataset)
    if args.max_samples is not None:
        eval_samples = eval_samples[: args.max_samples]

    answerable_samples = [s for s in eval_samples if s.get("behavior") != "not_found"]
    not_found_samples = [s for s in eval_samples if s.get("behavior") == "not_found"]

    print(
        f"[INFO] Total sampel: {len(eval_samples)} | "
        f"answerable: {len(answerable_samples)} | not_found: {len(not_found_samples)}"
    )

    rows = build_prediction_rows(
        samples=answerable_samples,
        model_name=args.model,
        top_k=args.top_k,
        temperature=args.temperature,
        no_rag=args.no_rag,
        use_source_scope=not args.no_source_scope,
    )

    not_found_rows = []
    if not_found_samples:
        print(f"[INFO] Generate {len(not_found_samples)} sampel not_found (di luar penilaian RAGAS)...")
        not_found_rows = build_prediction_rows(
            samples=not_found_samples,
            model_name=args.model,
            top_k=args.top_k,
            temperature=args.temperature,
            no_rag=args.no_rag,
            use_source_scope=not args.no_source_scope,
            )

    dataset = to_ragas_dataset(rows)

    if args.judge_provider == "anthropic":
        if ChatAnthropic is None:
            print("[ERROR] langchain-anthropic tidak terinstall. Jalankan: pip install langchain-anthropic")
            sys.exit(1)
        evaluator_model_name = args.evaluator_model or "claude-sonnet-4-6"
        llm_model = ChatAnthropic(
            model=evaluator_model_name,
            temperature=args.temperature,
        )
    elif args.judge_provider == "openai":
        if ChatOpenAI is None:
            print("[ERROR] langchain-openai tidak terinstall. Jalankan: pip install langchain-openai")
            sys.exit(1)
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            print("[ERROR] OPENAI_API_KEY belum diset di file .env")
            sys.exit(1)
        evaluator_model_name = args.evaluator_model or "gpt-4o"
        llm_model = ChatOpenAI(
            model=evaluator_model_name,
            api_key=openai_api_key,
            temperature=args.temperature,
            seed=JUDGE_SEED,
        )
    elif args.judge_provider == "deepseek":
        if ChatOpenAI is None:
            print("[ERROR] langchain-openai tidak terinstall. Jalankan: pip install langchain-openai")
            sys.exit(1)
        deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not deepseek_api_key or deepseek_api_key == "your-deepseek-api-key-here":
            print("[ERROR] DEEPSEEK_API_KEY belum diset di file .env")
            sys.exit(1)
        evaluator_model_name = args.evaluator_model or "deepseek-chat"
        llm_model = ChatOpenAI(
            model=evaluator_model_name,
            api_key=deepseek_api_key,
            base_url="https://api.deepseek.com",
            temperature=args.temperature,
        )
    else:
        evaluator_model_name = args.evaluator_model or args.model
        ollama_base_url = re.sub(r"/api/generate$", "", args.ollama_url).rstrip("/") if args.ollama_url else None
        llm_model = ChatOllama(
            model=evaluator_model_name,
            temperature=args.temperature,
            format="json",
            **({"base_url": ollama_base_url} if ollama_base_url else {}),
        )

    if args.no_rag:
        active_metric_names = list(NO_RAG_METRIC_NAMES)
    else:
        active_metric_names = METRIC_NAMES + (CONTEXT_METRIC_NAMES if args.context_metrics else [])
    metrics = build_ragas_metrics(llm_model, include_context_metrics=args.context_metrics)

    run_config = RunConfig(
        timeout=600,
        max_retries=2,
        max_wait=120,
        max_workers=args.max_workers,
    )
    track_cost = args.judge_provider in ("openai", "anthropic")
    cost_token_parser = (
        get_token_usage_for_openai if args.judge_provider == "openai"
        else get_token_usage_for_anthropic if args.judge_provider == "anthropic"
        else None
    )
    metric_values = {}
    metric_errors = {}
    metric_coverage = {}
    total_input_tokens = 0
    total_output_tokens = 0
    print("\n  Evaluasi metrik:")
    for metric_name in active_metric_names:
        print(f"  - {metric_name}:")
        t0 = time.time()
        values, error, token_usage = evaluate_metric(
            dataset=dataset,
            metric_name=metric_name,
            metric=metrics[metric_name],
            run_config=run_config,
            token_usage_parser=cost_token_parser if track_cost else None,
        )
        elapsed = time.time() - t0
        metric_values[metric_name] = values
        metric_errors[metric_name] = error
        coverage = sum(1 for value in values if value is not None)
        metric_coverage[metric_name] = coverage
        if token_usage is not None:
            total_input_tokens += token_usage.input_tokens
            total_output_tokens += token_usage.output_tokens
        score = mean_ignore_none(values)
        mins, secs = divmod(int(elapsed), 60)
        score_text = f"{score:.4f}" if score is not None else "N/A"
        print(f"    -> {score_text}   ({coverage}/{len(dataset)} ok, {mins}m{secs:02d}s)")
        if error:
            print(f"    !! error: {error}")

    estimated_cost_usd = None
    if track_cost:
        pricing_table = OPENAI_PRICING if args.judge_provider == "openai" else ANTHROPIC_PRICING
        pricing = pricing_table.get(evaluator_model_name)
        if pricing:
            estimated_cost_usd = (
                total_input_tokens * pricing["input"] + total_output_tokens * pricing["output"]
            )
        else:
            print(f"[WARN] Harga untuk model '{evaluator_model_name}' belum ada di tabel harga, biaya tidak dihitung.")

    mode_counts = {}
    for row in rows:
        mode_counts[row["mode"]] = mode_counts.get(row["mode"], 0) + 1

    aggregate_scores = {}
    metric_coverage_scoped = {}
    for metric_name in active_metric_names:
        score = mean_ignore_none(metric_values[metric_name])
        if score is not None:
            aggregate_scores[metric_name] = score

    faithfulness_excluded_count = 0
    generation_coverage = {}
    if not args.no_rag:
        faithfulness_scoped_values = [
            value for row, value in zip(rows, metric_values["faithfulness"])
            if not is_out_of_faithfulness_scope(row)
        ]
        faithfulness_excluded_count = sum(
            1 for row in rows if is_out_of_faithfulness_scope(row)
        )
        aggregate_scores["faithfulness_all_modes"] = aggregate_scores.get("faithfulness")
        scoped_score = mean_ignore_none(faithfulness_scoped_values)
        if scoped_score is not None:
            aggregate_scores["faithfulness"] = scoped_score
        metric_coverage_scoped["faithfulness"] = sum(
            1 for value in faithfulness_scoped_values if value is not None
        )

        faithfulness_generated_values = [
            value for row, value in zip(rows, metric_values["faithfulness"])
            if not is_out_of_faithfulness_scope(row) and row["used_generation"]
        ]
        faithfulness_deterministic_values = [
            value for row, value in zip(rows, metric_values["faithfulness"])
            if not is_out_of_faithfulness_scope(row) and not row["used_generation"]
        ]
        faithfulness_generated_only = mean_ignore_none(faithfulness_generated_values)
        faithfulness_deterministic_only = mean_ignore_none(faithfulness_deterministic_values)
        if faithfulness_generated_only is not None:
            aggregate_scores["faithfulness_generated_only"] = faithfulness_generated_only
        if faithfulness_deterministic_only is not None:
            aggregate_scores["faithfulness_deterministic_only"] = faithfulness_deterministic_only
        generation_coverage = {
            "generated_only": sum(1 for v in faithfulness_generated_values if v is not None),
            "generated_only_n": len(faithfulness_generated_values),
            "deterministic_only": sum(1 for v in faithfulness_deterministic_values if v is not None),
            "deterministic_only_n": len(faithfulness_deterministic_values),
        }

        for scoped_metric_name in ("answer_relevancy", "context_precision"):
            if scoped_metric_name not in metric_values:
                continue
            scoped_values = [
                value for row, value in zip(rows, metric_values[scoped_metric_name])
                if not is_out_of_faithfulness_scope(row)
            ]
            scoped_metric_score = mean_ignore_none(scoped_values)
            if scoped_metric_score is None:
                continue
            aggregate_scores[f"{scoped_metric_name}_all_modes"] = aggregate_scores.get(scoped_metric_name)
            aggregate_scores[scoped_metric_name] = scoped_metric_score
            metric_coverage_scoped[scoped_metric_name] = sum(
                1 for value in scoped_values if value is not None
            )

    per_sample = []
    for index, row in enumerate(rows):
        sample_entry = {
            **row,
            "answer_relevancy": metric_values["answer_relevancy"][index],
            "semantic_similarity": metric_values["semantic_similarity"][index],
        }
        if not args.no_rag:
            sample_entry["faithfulness"] = metric_values["faithfulness"][index]
            sample_entry["faithfulness_in_scope"] = (
                not is_out_of_faithfulness_scope(row)
            )
        if args.context_metrics:
            sample_entry["context_precision"] = metric_values["context_precision"][index]
            sample_entry["context_recall"] = metric_values["context_recall"][index]
        per_sample.append(sample_entry)

    payload = {
        "run_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": str(Path(args.dataset).resolve()),
        "generator_model": args.model,
        "judge_provider": args.judge_provider,
        "evaluator_model": evaluator_model_name,
        "ollama_url": args.ollama_url,
        "top_k": args.top_k,
        "temperature": args.temperature,
        "generator_seed": _ask_rag.OLLAMA_SEED,
        "judge_seed": JUDGE_SEED,
        "sample_count": len(rows),
        "no_rag": args.no_rag,
        "context_metrics": args.context_metrics,
        "retrieval_condition": (
            "no_rag" if args.no_rag
            else "oracle" if any(r["retrieval_filters"].get("source_id") for r in rows)
            else "produksi"
        ),
        "mode_counts": mode_counts,
        "aggregate_scores": aggregate_scores,
        "faithfulness_scope_note": (
            "BASELINE NO-RAG: retrieval dilewati, contexts kosong. faithfulness dan "
            "context metrics TIDAK dihitung karena tidak terdefinisi tanpa konteks. "
            "Hanya answer_relevancy dan semantic_similarity yang berlaku."
            if args.no_rag else
            "faithfulness = rata-rata HANYA mode yang grounded ke context_items. "
            "Sampel dikecualikan bila mode-nya termasuk daftar berikut ATAU "
            "context_items-nya kosong (faithfulness tidak terdefinisi tanpa "
            "konteks -- alasan yang sama dipakai baseline --no-rag). "
            "Daftar mode: " + ", ".join(sorted(FAITHFULNESS_SCOPE_EXCLUDED_MODES)) + ". "
            "Semua mode itu adalah guardrail yang MEMANG SUDAH ADA di ask_rag.py; "
            "tidak ada logika penolakan baru yang ditambahkan demi evaluasi. "
            "faithfulness_all_modes = rata-rata semua mode termasuk yang dikecualikan, "
            "disimpan untuk transparansi/pembanding. "
            "faithfulness_generated_only = HANYA sampel used_generation=true (murni "
            "kemampuan generatif model, paling rawan halusinasi -- ini yang paling "
            "representatif untuk klaim kualitas generatif). "
            "faithfulness_deterministic_only = HANYA sampel used_generation=false "
            "(exact_fact/rag_direct, ekstraksi teks apa adanya -- wajar mendekati "
            "1.0, bukan indikasi kemampuan generatif). "
            "answer_relevancy dan context_precision menerapkan scoping yang SAMA "
            "(exclude mode yang sama) dengan alasan yang sama: kedua metrik ini "
            "tidak terdefinisi secara wajar untuk jawaban template/penolakan yang "
            "tidak digenerate dari konteks. *_all_modes untuk keduanya juga "
            "disimpan untuk transparansi/pembanding."
        ),
        "faithfulness_excluded_count": faithfulness_excluded_count,
        "generation_coverage": generation_coverage,
        "metric_coverage": metric_coverage,
        "metric_errors": metric_errors,
        "token_usage": (
            {"input_tokens": total_input_tokens, "output_tokens": total_output_tokens}
            if track_cost else None
        ),
        "estimated_cost_usd": estimated_cost_usd,
        "results": per_sample,
        "not_found_count": len(not_found_rows),
        "not_found_results": [
            {
                "id": row["id"],
                "question": row["question"],
                "mode": row["mode"],
                "answer": row["answer"],
                "reference_answer": row["reference_answer"],
            }
            for row in not_found_rows
        ],
    }


    saved_path, save_warning = write_results(args.output, payload)

    print(f"\n[DONE] Dataset         : {Path(args.dataset).name}")
    print(f"[DONE] Model generator : {args.model} (Ollama)")
    print(f"[DONE] Model judge     : {evaluator_model_name} ({args.judge_provider})")
    print(f"[DONE] Sampel dinilai  : {len(rows)} answerable, {len(not_found_rows)} not_found (di luar penilaian)")
    print(f"[DONE] Kondisi retrieval: {payload['retrieval_condition']}")
    for metric_name in active_metric_names:
        score = aggregate_scores.get(metric_name)
        if metric_name == "faithfulness":
            cov = metric_coverage_scoped.get("faithfulness", 0)
        else:
            cov = metric_coverage.get(metric_name, 0)
        total_for_metric = len(rows)
        score_text = f"{score:.4f}" if score is not None else "N/A"
        print(f"[DONE] {metric_name:20s} : {score_text}  (n={cov}/{total_for_metric})")
        if metric_errors.get(metric_name):
            print(f"       !! error: {metric_errors[metric_name]}")
    if track_cost:
        cost_text = f"${estimated_cost_usd:.4f}" if estimated_cost_usd is not None else "tidak dihitung"
        print(f"[DONE] Biaya judge     : {cost_text}  ({total_input_tokens:,} in / {total_output_tokens:,} out token)")
    if save_warning:
        print(f"[WARNING] {save_warning}")
    print(f"[DONE] Hasil detail    : {saved_path}")


if __name__ == "__main__":
    main()