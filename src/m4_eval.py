from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""

    zeros = {"faithfulness": 0.0, "answer_relevancy": 0.0,
             "context_precision": 0.0, "context_recall": 0.0, "per_question": []}
    if not questions:
        return zeros
    try:
        import asyncio
        from config import OPENAI_API_KEY
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from ragas.run_config import RunConfig
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        eval_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, request_timeout=30, max_retries=1, api_key=OPENAI_API_KEY)
        eval_embeddings = OpenAIEmbeddings(model="text-embedding-3-small", request_timeout=30, api_key=OPENAI_API_KEY)
        wrapped_llm = LangchainLLMWrapper(eval_llm)
        wrapped_embeddings = LangchainEmbeddingsWrapper(eval_embeddings)

        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        for m in metrics:
            m.llm = wrapped_llm
            if hasattr(m, "embeddings"):
                m.embeddings = wrapped_embeddings
            m.init(RunConfig())

        async def _eval_one(q, a, ctx, gt, idx, total):
            row = {"question": q, "answer": a, "contexts": ctx, "ground_truth": gt}
            f, cp, cr, ar = await asyncio.gather(
                faithfulness.ascore(row),
                context_precision.ascore(row),
                context_recall.ascore(row),
                answer_relevancy.ascore(row),
                return_exceptions=True
            )
            def _clean(val):
                return 0.0 if isinstance(val, Exception) or val != val else float(val)

            f_score = _clean(f)
            cp_score = _clean(cp)
            cr_score = _clean(cr)
            ar_score = _clean(ar)
            print(f"  [{idx+1}/{total}] Faith: {f_score:.2f} | C-Prec: {cp_score:.2f} | C-Rec: {cr_score:.2f} | Relev: {ar_score:.2f}", flush=True)

            return EvalResult(
                question=q, answer=a, contexts=list(ctx), ground_truth=gt,
                faithfulness=f_score,
                answer_relevancy=ar_score,
                context_precision=cp_score,
                context_recall=cr_score,
            )

        async def _eval_all():
            tasks = [
                _eval_one(q, a, ctx, gt, i, len(questions))
                for i, (q, a, ctx, gt) in enumerate(zip(questions, answers, contexts, ground_truths))
            ]
            # Chạy song song từng đợt 4 câu để vừa nhanh vừa không quá tải API
            sem = asyncio.Semaphore(4)
            async def _throttled(coro):
                async with sem:
                    return await coro
            return await asyncio.gather(*[_throttled(t) for t in tasks])

        per_question = asyncio.run(_eval_all())

        n = max(len(per_question), 1)
        return {
            "faithfulness": sum(r.faithfulness for r in per_question) / n,
            "answer_relevancy": sum(r.answer_relevancy for r in per_question) / n,
            "context_precision": sum(r.context_precision for r in per_question) / n,
            "context_recall": sum(r.context_recall for r in per_question) / n,
            "per_question": per_question,
        }
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return zeros


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }

    scored = []
    for r in eval_results:
        metrics = {m: getattr(r, m) for m in diagnostic_tree}
        avg = sum(metrics.values()) / len(metrics)
        worst_metric = min(metrics, key=metrics.get)
        scored.append((avg, worst_metric, metrics[worst_metric], r))

    scored.sort(key=lambda x: x[0])
    failures = []
    for avg, worst_metric, score, r in scored[:bottom_n]:
        diagnosis, fix = diagnostic_tree[worst_metric]
        failures.append({
            "question": r.question,
            "worst_metric": worst_metric,
            "score": round(score, 4),
            "avg_score": round(avg, 4),
            "diagnosis": diagnosis,
            "suggested_fix": fix,
        })
    return failures


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
