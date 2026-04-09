import json
from pathlib import Path
from typing import List
from rouge_score import rouge_scorer
from schemas import EvalQuestion, EvalResult, EvalReport
from retriever import hybrid_retrieve
from generator import generate_answer


def simple_faithfulness(answer: str, chunks: List[dict]) -> float:
    """
    Rough faithfulness: fraction of answer sentences
    that have at least one matching phrase in retrieved context.
    """
    if not chunks:
        return 0.0

    context = " ".join([c["text"].lower() for c in chunks])
    sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]

    if not sentences:
        return 0.0

    matched = sum(
        1 for s in sentences
        if any(word in context for word in s.lower().split() if len(word) > 4)
    )
    return round(matched / len(sentences), 3)


def evaluate(questions: List[EvalQuestion], collection_name: str = "default") -> EvalReport:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    results = []

    for eq in questions:
        chunks = hybrid_retrieve(eq.question, collection_name=eq.collection)
        answer = generate_answer(eq.question, chunks)

        rouge_l = scorer.score(eq.ground_truth, answer)["rougeL"].fmeasure
        faithfulness = simple_faithfulness(answer, chunks)

        results.append(EvalResult(
            question=eq.question,
            answer=answer,
            ground_truth=eq.ground_truth,
            rouge_l=round(rouge_l, 3),
            faithfulness_score=faithfulness,
        ))

    avg_rouge = sum(r.rouge_l for r in results) / len(results) if results else 0
    avg_faith = sum(r.faithfulness_score for r in results) / len(results) if results else 0

    return EvalReport(
        total=len(results),
        avg_rouge_l=round(avg_rouge, 3),
        avg_faithfulness=round(avg_faith, 3),
        results=results,
    )