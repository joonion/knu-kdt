from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_qna.documents import build_chunks
from rag_qna.evaluator import DEFAULT_BERTSCORE_MODEL, DEFAULT_RAGAS_METRICS, evaluate
from rag_qna.generator import DEFAULT_LLM_MODEL, HuggingFaceAnswerGenerator
from rag_qna.index import DEFAULT_EMBEDDING_MODEL, VectorIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG QnA with Hugging Face LLMs and BERTScore")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build-index", help="Build a vector index from documents")
    build_parser.add_argument("--docs", type=Path, required=True, help="Directory containing .md/.txt documents")
    build_parser.add_argument("--index", type=Path, required=True, help="Directory where index files are saved")
    build_parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    build_parser.add_argument("--chunk-size", type=int, default=700)
    build_parser.add_argument("--overlap", type=int, default=120)

    ask_parser = subparsers.add_parser("ask", help="Ask a question against an existing index")
    ask_parser.add_argument("--index", type=Path, required=True, help="Directory containing index files")
    ask_parser.add_argument("--question", required=True)
    ask_parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    ask_parser.add_argument("--top-k", type=int, default=4)
    ask_parser.add_argument("--show-contexts", action="store_true")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate answers with BERTScore and optional RAGAS metrics")
    eval_parser.add_argument("--index", type=Path, required=True, help="Directory containing index files")
    eval_parser.add_argument("--eval-file", type=Path, required=True, help="JSONL file with question/reference rows")
    eval_parser.add_argument("--output", type=Path, default=Path("artifacts/eval_results.json"))
    eval_parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    eval_parser.add_argument("--bertscore-model", default=DEFAULT_BERTSCORE_MODEL)
    eval_parser.add_argument("--top-k", type=int, default=4)
    eval_parser.add_argument("--with-ragas", action="store_true", help="Also run RAGAS RAG evaluation metrics")
    eval_parser.add_argument(
        "--ragas-metrics",
        default=",".join(DEFAULT_RAGAS_METRICS),
        help="Comma-separated RAGAS metrics: faithfulness,context_recall,context_precision,factual_correctness,response_relevancy",
    )
    eval_parser.add_argument("--ragas-llm-model", default="gpt-4o-mini", help="Evaluator LLM name passed to ragas.llm_factory")
    eval_parser.add_argument("--ragas-llm-provider", default=None, help="Optional ragas llm_factory provider, e.g. openai, google, ollama")
    eval_parser.add_argument("--ragas-base-url", default=None, help="Optional OpenAI-compatible or local evaluator endpoint")
    eval_parser.add_argument("--ragas-embedding-provider", default=None, help="Required only for response_relevancy")
    eval_parser.add_argument("--ragas-embedding-model", default=None, help="Optional evaluator embedding model")

    args = parser.parse_args()

    if args.command == "build-index":
        chunks = build_chunks(args.docs, chunk_size=args.chunk_size, overlap=args.overlap)
        index = VectorIndex.from_chunks(chunks, model_name=args.embedding_model)
        index.save(args.index)
        print(f"Saved {len(chunks)} chunks to {args.index}")
        return

    if args.command == "ask":
        index = VectorIndex.load(args.index)
        generator = HuggingFaceAnswerGenerator(model_name=args.llm_model)
        result = generator.answer(args.question, index.search(args.question, top_k=args.top_k))
        print(result.answer)
        if args.show_contexts:
            print(json.dumps(result.contexts, ensure_ascii=False, indent=2))
        return

    if args.command == "evaluate":
        index = VectorIndex.load(args.index)
        generator = HuggingFaceAnswerGenerator(model_name=args.llm_model)
        result = evaluate(
            index=index,
            generator=generator,
            eval_file=args.eval_file,
            top_k=args.top_k,
            bertscore_model=args.bertscore_model,
            with_ragas=args.with_ragas,
            ragas_metrics=tuple(name.strip() for name in args.ragas_metrics.split(",") if name.strip()),
            ragas_llm_model=args.ragas_llm_model,
            ragas_llm_provider=args.ragas_llm_provider,
            ragas_base_url=args.ragas_base_url,
            ragas_embedding_provider=args.ragas_embedding_provider,
            ragas_embedding_model=args.ragas_embedding_model,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
        print(f"Saved evaluation results to {args.output}")
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
