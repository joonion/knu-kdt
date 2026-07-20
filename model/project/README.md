# Hugging Face LLM RAG-QnA with BERTScore

Hugging Face 모델을 사용해 문서 기반 RAG Question Answering을 수행하고, 생성 답변을 BERTScore와 RAGAS 지표로 평가하는 예제 프로젝트입니다.

## 주요 기능

- Markdown/Text 문서를 청크로 분할
- Sentence-Transformers 임베딩으로 검색 인덱스 생성
- Hugging Face `transformers` LLM으로 근거 기반 답변 생성
- JSONL 평가셋에 대해 Precision/Recall/F1 BERTScore 산출
- RAGAS로 faithfulness, context recall, context precision, factual correctness 평가
- 검색 결과와 생성 프롬프트를 확인할 수 있는 CLI 제공

## 설치

Python 3.10 이상을 권장합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

GPU 환경에서는 PyTorch 설치가 시스템 CUDA 버전에 따라 달라질 수 있습니다. 필요한 경우 [PyTorch 공식 설치 안내](https://pytorch.org/get-started/locally/)에 맞춰 `torch`를 먼저 설치한 뒤 `requirements.txt`를 설치하세요.

## 빠른 실행

샘플 문서로 검색 인덱스를 만듭니다.

```powershell
python -m rag_qna.cli build-index --docs data/docs --index artifacts/index
```

질문에 답변합니다.

```powershell
python -m rag_qna.cli ask --index artifacts/index --question "RAG는 어떤 단계로 답변을 생성하나요?"
```

평가셋으로 BERTScore를 계산합니다.

```powershell
python -m rag_qna.cli evaluate --index artifacts/index --eval-file data/eval/sample_qa.jsonl --output artifacts/eval_results.json
```

RAGAS 지표까지 함께 계산하려면 evaluator LLM을 사용할 수 있는 API 키나 로컬 엔드포인트가 필요합니다.

```powershell
python -m rag_qna.cli evaluate `
  --index artifacts/index `
  --eval-file data/eval/sample_qa.jsonl `
  --output artifacts/eval_results.json `
  --with-ragas `
  --ragas-llm-model "gpt-4o-mini"
```

Ollama처럼 로컬 evaluator를 쓰는 경우에는 provider와 base URL을 지정합니다.

```powershell
python -m rag_qna.cli evaluate `
  --index artifacts/index `
  --eval-file data/eval/sample_qa.jsonl `
  --with-ragas `
  --ragas-llm-provider ollama `
  --ragas-llm-model mistral `
  --ragas-base-url http://localhost:11434
```

## 모델 변경

기본 모델은 로컬/CPU에서도 가볍게 테스트할 수 있도록 작은 모델로 설정되어 있습니다.

- 임베딩 모델: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- 생성 모델: `google/flan-t5-small`
- BERTScore 모델: `bert-base-multilingual-cased`
- RAGAS evaluator 모델: `gpt-4o-mini` 또는 `ollama`, OpenAI 호환 엔드포인트 등

더 좋은 답변 품질이 필요하면 `--llm-model`에 instruction-tuned Seq2Seq 또는 Causal LM 모델을 지정할 수 있습니다.

```powershell
python -m rag_qna.cli ask `
  --index artifacts/index `
  --question "내 질문" `
  --llm-model "google/flan-t5-base"
```

## 평가셋 형식

평가 파일은 JSONL 형식이며 한 줄에 하나의 샘플을 둡니다.

```json
{"question": "RAG란 무엇인가요?", "reference": "RAG는 검색한 외부 문맥을 활용해 답변을 생성하는 방식입니다."}
```

선택적으로 `id`를 포함할 수 있습니다.

## RAGAS 지표

기본 RAGAS 지표는 다음 네 가지입니다.

- `faithfulness`: 생성 답변이 검색 문맥에 의해 뒷받침되는지 평가
- `context_recall`: 정답에 필요한 정보가 검색 문맥에 포함되었는지 평가
- `context_precision`: 검색 문맥의 상위 랭킹이 정답과 관련 있는지 평가
- `factual_correctness`: 생성 답변과 기준 답변의 사실 일치도를 평가

`--ragas-metrics`로 실행할 지표를 쉼표로 지정할 수 있습니다.

```powershell
python -m rag_qna.cli evaluate `
  --index artifacts/index `
  --eval-file data/eval/sample_qa.jsonl `
  --with-ragas `
  --ragas-metrics faithfulness,context_recall,context_precision
```

`response_relevancy`를 추가하려면 evaluator embedding 설정도 필요합니다.

```powershell
python -m rag_qna.cli evaluate `
  --index artifacts/index `
  --eval-file data/eval/sample_qa.jsonl `
  --with-ragas `
  --ragas-metrics faithfulness,response_relevancy `
  --ragas-embedding-provider openai `
  --ragas-embedding-model text-embedding-3-small
```

## 프로젝트 구조

```text
.
├── data/
│   ├── docs/              # 검색 대상 문서
│   └── eval/              # 평가용 JSONL
├── src/rag_qna/
│   ├── cli.py             # 명령행 인터페이스
│   ├── documents.py       # 문서 로딩 및 청크 분할
│   ├── evaluator.py       # BERTScore 및 RAGAS 평가
│   ├── generator.py       # Hugging Face LLM 답변 생성
│   └── index.py           # 임베딩 인덱스 생성/검색
└── requirements.txt
```
