"""Evaluation harness for the codebase Q&A agent.

Layer 1 (no AWS needed): retrieval quality. Did the right file show up in the top k results?
Layer 2 (needs Bedrock access): answer quality. Does the agent's answer contain the expected
facts, and does it refuse to invent answers for questions the code cannot answer?

Usage:
    python eval/run_eval.py                 # retrieval only
    python eval/run_eval.py --answers       # also invokes the agent through Bedrock
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retriever import CodeIndex  # noqa: E402


def retrieval_eval(index: CodeIndex, cases: list[dict], k: int) -> dict:
    hits, reciprocal_ranks, rows = 0, [], []
    answerable = [c for c in cases if c["expected_files"]]
    for case in answerable:
        results = index.search(case["question"], top_k=k)
        ranked_files = [chunk.path for _, chunk in results]
        first_rank = next(
            (i + 1 for i, path in enumerate(ranked_files)
             if any(expected in path for expected in case["expected_files"])),
            None,
        )
        hit = first_rank is not None
        hits += hit
        reciprocal_ranks.append(1 / first_rank if first_rank else 0.0)
        rows.append((case["id"], hit, first_rank, case["question"]))
    return {
        "recall_at_k": hits / len(answerable),
        "mrr": sum(reciprocal_ranks) / len(answerable),
        "k": k,
        "n": len(answerable),
        "rows": rows,
    }


def answer_eval(cases: list[dict]) -> dict:
    from agent import ask  # imported lazily so retrieval eval works without the AWS SDKs

    passed, rows = 0, []
    for case in cases:
        answer = ask(case["question"]).lower()
        facts = [f.lower() for f in case["expected_facts"]]
        if case["expected_files"]:
            ok = all(f in answer for f in facts)
        else:
            # unanswerable case: accept any clear "not in the code" style refusal
            ok = any(p in answer for p in ["not found", "does not contain", "no evidence", "not in the"])
        passed += ok
        rows.append((case["id"], ok, case["question"]))
    return {"accuracy": passed / len(cases), "n": len(cases), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", action="store_true", help="also score agent answers via Bedrock")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.0,
        help="exit with code 1 if retrieval recall@k is below this value (used by CI)",
    )
    args = parser.parse_args()

    cases = json.loads((ROOT / "eval" / "golden_set.json").read_text())
    index = CodeIndex(ROOT / "sample_repo")

    r = retrieval_eval(index, cases, args.k)
    print(f"Retrieval: recall@{r['k']} = {r['recall_at_k']:.0%}, MRR = {r['mrr']:.2f} over {r['n']} questions")
    for qid, hit, rank, question in r["rows"]:
        print(f"  {'PASS' if hit else 'MISS'}  {qid}  rank={rank}  {question}")

    if r["recall_at_k"] < args.min_recall:
        print(f"\nFAIL: recall@{r['k']} {r['recall_at_k']:.0%} is below the required {args.min_recall:.0%}")
        sys.exit(1)

    if args.answers:
        a = answer_eval(cases)
        print(f"\nAnswers: accuracy = {a['accuracy']:.0%} over {a['n']} questions")
        for qid, ok, question in a["rows"]:
            print(f"  {'PASS' if ok else 'FAIL'}  {qid}  {question}")


if __name__ == "__main__":
    main()
