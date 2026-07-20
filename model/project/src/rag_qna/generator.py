from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from rag_qna.documents import DocumentChunk


DEFAULT_LLM_MODEL = "google/flan-t5-small"


@dataclass(frozen=True)
class GeneratedAnswer:
    question: str
    answer: str
    contexts: list[dict[str, object]]
    prompt: str


class HuggingFaceAnswerGenerator:
    def __init__(self, model_name: str = DEFAULT_LLM_MODEL, max_new_tokens: int = 192):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        config = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if config.is_encoder_decoder:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.is_encoder_decoder = True
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.is_encoder_decoder = False
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()

    def answer(self, question: str, contexts: list[tuple[DocumentChunk, float]]) -> GeneratedAnswer:
        prompt = build_prompt(question, [chunk for chunk, _ in contexts])
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        if self.is_encoder_decoder:
            answer_ids = output_ids[0]
        else:
            answer_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        answer_text = self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        return GeneratedAnswer(
            question=question,
            answer=answer_text,
            contexts=[
                {"id": chunk.id, "source": chunk.source, "score": score, "text": chunk.text}
                for chunk, score in contexts
            ],
            prompt=prompt,
        )


def build_prompt(question: str, contexts: list[DocumentChunk]) -> str:
    context_text = "\n\n".join(
        f"[Context {index + 1} | {chunk.source}]\n{chunk.text}" for index, chunk in enumerate(contexts)
    )
    return (
        "Answer the question using only the provided context. "
        "If the context is insufficient, say that the answer is not available in the context.\n\n"
        f"{context_text}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
