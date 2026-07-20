from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from rag_qna.generator import HuggingFaceAnswerGenerator
from rag_qna.index import VectorIndex


DEFAULT_BERTSCORE_MODEL = "bert-base-multilingual-cased"
DEFAULT_RAGAS_METRICS = ("faithfulness", "context_recall", "context_precision", "factual_correctness")


@dataclass(frozen=True)
class EvaluationRow:
    id: str
    question: str
    reference: str
    prediction: str
    bertscore_precision: float
    bertscore_recall: float
    bertscore_f1: float


def load_eval_file(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if "question" not in row or "reference" not in row:
            raise ValueError(f"Line {line_number} must contain question and reference fields")
        rows.append(
            {
                "id": str(row.get("id", line_number)),
                "question": str(row["question"]),
                "reference": str(row["reference"]),
            }
        )

    if not rows:
        raise ValueError(f"Evaluation file is empty: {path}")

    return rows


def evaluate(
    index: VectorIndex,
    generator: HuggingFaceAnswerGenerator,
    eval_file: Path,
    top_k: int = 4,
    bertscore_model: str = DEFAULT_BERTSCORE_MODEL,
    with_ragas: bool = False,
    ragas_metrics: tuple[str, ...] = DEFAULT_RAGAS_METRICS,
    ragas_llm_model: str = "gpt-4o-mini",
    ragas_llm_provider: str | None = None,
    ragas_base_url: str | None = None,
    ragas_embedding_provider: str | None = None,
    ragas_embedding_model: str | None = None,
) -> dict[str, object]:
    examples = load_eval_file(eval_file)
    predictions: list[str] = []
    references: list[str] = []
    ragas_records: list[dict[str, object]] = []

    for example in examples:
        contexts = index.search(example["question"], top_k=top_k)
        generated = generator.answer(example["question"], contexts)
        predictions.append(generated.answer)
        references.append(example["reference"])
        ragas_records.append(
            {
                "user_input": example["question"],
                "response": generated.answer,
                "retrieved_contexts": [context["text"] for context in generated.contexts],
                "reference": example["reference"],
            }
        )

    precision, recall, f1, bertscore_implementation = calculate_bertscore(
        predictions=predictions,
        references=references,
        model_name=bertscore_model,
    )

    rows = [
        EvaluationRow(
            id=example["id"],
            question=example["question"],
            reference=example["reference"],
            prediction=prediction,
            bertscore_precision=float(p),
            bertscore_recall=float(r),
            bertscore_f1=float(f),
        )
        for example, prediction, p, r, f in zip(examples, predictions, precision, recall, f1, strict=True)
    ]

    result: dict[str, object] = {
        "metrics": {
            "bertscore_precision": mean(row.bertscore_precision for row in rows),
            "bertscore_recall": mean(row.bertscore_recall for row in rows),
            "bertscore_f1": mean(row.bertscore_f1 for row in rows),
        },
        "metadata": {"bertscore_implementation": bertscore_implementation},
        "rows": [asdict(row) for row in rows],
    }

    if with_ragas:
        ragas_result = evaluate_with_ragas(
            records=ragas_records,
            metric_names=ragas_metrics,
            llm_model=ragas_llm_model,
            llm_provider=ragas_llm_provider,
            base_url=ragas_base_url,
            embedding_provider=ragas_embedding_provider,
            embedding_model=ragas_embedding_model,
        )
        result["metrics"].update({f"ragas_{key}": value for key, value in ragas_result["metrics"].items()})
        result["ragas"] = ragas_result

    return result


def calculate_bertscore(
    predictions: list[str],
    references: list[str],
    model_name: str,
) -> tuple[list[float], list[float], list[float], str]:
    try:
        from bert_score import score as bert_score

        precision, recall, f1 = bert_score(
            predictions,
            references,
            model_type=model_name,
            lang="ko",
            verbose=True,
        )
        return (
            [float(value) for value in precision],
            [float(value) for value in recall],
            [float(value) for value in f1],
            "bert-score",
        )
    except (AttributeError, ImportError, TypeError) as exc:
        print(f"Falling back to local BERTScore calculation because bert-score failed: {exc}")
        precision, recall, f1 = calculate_local_bertscore(predictions, references, model_name=model_name)
        return precision, recall, f1, "local-token-cosine"


def calculate_local_bertscore(
    predictions: list[str],
    references: list[str],
    model_name: str,
    max_length: int = 512,
) -> tuple[list[float], list[float], list[float]]:
    import torch
    import torch.nn.functional as functional
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    precision_scores: list[float] = []
    recall_scores: list[float] = []
    f1_scores: list[float] = []

    for prediction, reference in zip(predictions, references, strict=True):
        if not prediction.strip() or not reference.strip():
            precision_scores.append(0.0)
            recall_scores.append(0.0)
            f1_scores.append(0.0)
            continue

        prediction_embedding = _sentence_token_embeddings(tokenizer, model, prediction, max_length=max_length)
        reference_embedding = _sentence_token_embeddings(tokenizer, model, reference, max_length=max_length)

        if prediction_embedding.numel() == 0 or reference_embedding.numel() == 0:
            precision_scores.append(0.0)
            recall_scores.append(0.0)
            f1_scores.append(0.0)
            continue

        similarity = functional.normalize(prediction_embedding, dim=-1) @ functional.normalize(
            reference_embedding, dim=-1
        ).T
        precision = float(similarity.max(dim=1).values.mean())
        recall = float(similarity.max(dim=0).values.mean())
        f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    return precision_scores, recall_scores, f1_scores


def _sentence_token_embeddings(tokenizer: object, model: object, text: str, max_length: int):
    import torch

    inputs = tokenizer(
        text,
        return_tensors="pt",
        return_special_tokens_mask=True,
        truncation=True,
        max_length=max_length,
    )
    special_tokens_mask = inputs.pop("special_tokens_mask").bool()[0]
    with torch.no_grad():
        outputs = model(**inputs)
    token_embeddings = outputs.last_hidden_state[0]
    valid_tokens = inputs["attention_mask"].bool()[0] & ~special_tokens_mask
    return token_embeddings[valid_tokens]


def evaluate_with_ragas(
    records: list[dict[str, object]],
    metric_names: tuple[str, ...],
    llm_model: str,
    llm_provider: str | None = None,
    base_url: str | None = None,
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
) -> dict[str, object]:
    """Run optional RAGAS metrics using the current RAGAS evaluate API."""
    _install_ragas_langchain_compat()
    try:
        from ragas import EvaluationDataset
        from ragas import evaluate as ragas_evaluate
        from ragas.llms import llm_factory
        try:
            from ragas.metrics.collections import (
                Faithfulness,
                FactualCorrectness,
                LLMContextPrecisionWithReference,
                LLMContextRecall,
                ResponseRelevancy,
            )
        except ImportError:
            from ragas.metrics import (
                Faithfulness,
                FactualCorrectness,
                LLMContextPrecisionWithReference,
                LLMContextRecall,
                ResponseRelevancy,
            )
    except ImportError as exc:
        raise RuntimeError(
            "RAGAS or one of its evaluator dependencies could not be imported. "
            "Run `pip install -e .` and ensure the LangChain provider packages required by RAGAS are installed. "
            f"Original error: {exc}"
        ) from exc

    llm_kwargs: dict[str, Any] = {}
    if llm_provider:
        llm_kwargs["provider"] = llm_provider
    if base_url:
        llm_kwargs["base_url"] = base_url
    evaluator_llm = llm_factory(llm_model, **llm_kwargs)

    evaluator_embeddings = None
    if "response_relevancy" in metric_names:
        evaluator_embeddings = _build_ragas_embeddings(embedding_provider, embedding_model)

    metric_builders = {
        "faithfulness": lambda: Faithfulness(llm=evaluator_llm),
        "context_recall": lambda: LLMContextRecall(llm=evaluator_llm),
        "context_precision": lambda: LLMContextPrecisionWithReference(llm=evaluator_llm),
        "factual_correctness": lambda: FactualCorrectness(llm=evaluator_llm),
        "response_relevancy": lambda: ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
    }

    unknown_metrics = sorted(set(metric_names) - set(metric_builders))
    if unknown_metrics:
        supported = ", ".join(sorted(metric_builders))
        raise ValueError(f"Unsupported RAGAS metrics: {', '.join(unknown_metrics)}. Supported metrics: {supported}")

    dataset = EvaluationDataset.from_list(records)
    result = ragas_evaluate(dataset=dataset, metrics=[metric_builders[name]() for name in metric_names], llm=evaluator_llm)
    rows = _ragas_rows(result)
    return {"metrics": _ragas_metrics(result, rows), "rows": rows}


def _install_ragas_langchain_compat() -> None:
    """Bridge RAGAS imports that still reference old LangChain provider paths."""
    import sys
    import types

    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return

    try:
        from langchain_google_vertexai import ChatVertexAI
    except ImportError:
        return

    vertexai_module = types.ModuleType(module_name)
    vertexai_module.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = vertexai_module


def _build_ragas_embeddings(provider: str | None, model: str | None) -> object:
    if not provider:
        raise ValueError(
            "RAGAS response_relevancy requires evaluator embeddings. "
            "Set --ragas-embedding-provider and optionally --ragas-embedding-model."
        )
    try:
        from ragas.embeddings.base import embedding_factory
    except ImportError as exc:
        raise RuntimeError("Installed RAGAS version does not expose embedding_factory.") from exc

    kwargs: dict[str, Any] = {}
    if model:
        kwargs["model"] = model
    return embedding_factory(provider, **kwargs)


def _ragas_rows(result: object) -> list[dict[str, object]]:
    if hasattr(result, "to_pandas"):
        return _jsonable(result.to_pandas().to_dict(orient="records"))
    return []


def _ragas_metrics(result: object, rows: list[dict[str, object]]) -> dict[str, float]:
    if isinstance(result, dict):
        return {str(key): float(value) for key, value in result.items() if isinstance(value, int | float)}
    if rows:
        metric_names = [key for key in rows[0] if key not in {"user_input", "response", "retrieved_contexts", "reference"}]
        return {
            name: mean(float(row[name]) for row in rows if isinstance(row.get(name), int | float))
            for name in metric_names
            if any(isinstance(row.get(name), int | float) for row in rows)
        }
    return {}


def _jsonable(value: object) -> object:
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "item"):
        return value.item()
    return value
