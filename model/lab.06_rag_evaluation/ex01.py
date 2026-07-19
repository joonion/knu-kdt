# ex01.py
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision
)

questions = [
    "아진산업의 최초 설립일은 언제인가요?",
    "아진산업의 2025년 매출액은 얼마인가요?",
    "아진산업의 본사 위치는 어디인가요?",
]

ground_truths = [
    "아진산업의 최초 설립일은 1978년 5월 31일입니다.",
    "아진산업의 2025년 매출액은 1조 88억 6783만원입니다.",
    "아진산업의 본사 위치는 경상북도 경산시 진량읍입니다.",
]

contexts = [
    ["아진산업은 1978년 5월 31일, 경상북도 경산시 진량읍에서 창립했으며,"],
    ["2024년에 7786억원, 2025년에 1조 89억원의 매출 실적을 달성했습니다."],
    ["아진산업은 1978년 5월 31일, 경상북도 경산시 진량읍에서 창립했으며,"],
]

answers_correct = [
    "아진산업은 1978년에 설립한 것으로 알려져 있습니다.",
    "2025년 아진산업은 총 1조 89억원의 매출을 달성했습니다.",
    "아진산업의 본사는 경상북도 경산시 진량읍에 위치하고 있습니다.",
]

answers_wrong = [
    "문맥 정보로는 아진산업의 창립일을 알 수 없습니다.",
    "문맥에 정보가 없으므로 2025년 아진산업 매출액은 모르겠습니다.",
    "아진산업의 본사 위치를 알 수 있는 문맥 정보가 없어서 모릅니다.",
]

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 평가용 LLM / Embedding 명시
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

for name, answers in [
    ("correct", answers_correct),
    ("wrong", answers_wrong),
]:
    dataset = Dataset.from_dict({
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": ground_truths
    })

    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=llm,
        embeddings=embeddings,
    )

    print(f"\n=== {name} ===")
    print(result)

    # df = result.to_pandas()
    # print(df)
