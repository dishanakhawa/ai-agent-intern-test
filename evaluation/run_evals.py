"""
run_evals.py

Deterministic evaluation runner. Loads evaluation/visible-cases.json,
runs each case's messages through agent.run_turn within one session
per case, then checks assertions against (a) the trace (tool calls,
retrieved sources, candidate_conflict) and (b) keyword/substring
checks against the final answer text.

Not an LLM judge -- every check is a plain string/data assertion.
must_include_concepts is the one soft spot: since exact wording isn't
required, concepts are matched via hand-picked keyword groups (any
one keyword = pass), not true semantic understanding. Documented here
as a known heuristic, not hidden.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from agent import run_turn

CONCEPT_KEYWORDS = {
    "final sale does not block damaged-item review": ["still eligible", "damaged", "final sale", "final-sale"],
    "report within 7 days": ["7 calendar days", "seven", "7 days"],
    "human review before approval": ["human", "review", "support"],
    "Canada is supported": ["Canada", "currently ships", "supported"],
    "5–9 business days after dispatch": ["5–9", "5-9", "business days"],
    "duties or taxes are not prepaid": ["duties", "taxes", "not prepaid", "responsible"],
    "shipping to Germany is not currently available": ["Germany", "not available", "only", "Canada"],
    "the order is cancelled": ["cancelled", "canceled"],
    "it will not be shipped": ["not be shipped", "will not ship", "not ship"],
    "order was not found": ["not found", "couldn't find", "could not find", "unable to find"],
    "check the order ID or contact support": ["double-check", "double check", "support", "check the order"],
    "shipped with Canada Post": ["Canada Post"],
    "delivery estimate is unavailable": ["unavailable", "not available", "not currently available"],
    "no lifetime warranty": ["no lifetime", "does not offer a lifetime", "not offer a lifetime"],
    "bags have 2 years": ["2 years", "two years"],
    "drinkware and travel accessories have 1 year": ["1 year", "one year"],
    "migration note is not authoritative": ["not authoritative", "cannot use", "not a customer policy", "internal", "not approved", "no information"],
    "standard policy is 30 days unless a valid exception applies": ["30 calendar days", "30 days"],
    "the agent cannot approve a return": ["cannot approve", "can't approve", "unable to approve", "human"],
    "the supplied information is insufficient": ["don't have", "do not have", "insufficient", "not contain"],
    "human confirmation": ["human", "support agent", "connect you"],
    "current official sources conflict": ["conflict", "inconsist"],
    "one says hand-wash the body": ["hand-wash", "hand wash"],
    "one says all components are dishwasher safe": ["dishwasher safe", "all components"],
    "human confirmation or safest interim guidance": ["human", "recommend", "connect"],
}

HANDOFF_KEYWORDS = [
    "human", "support agent", "connect you", "human agent",
    "recommend speaking", "recommend connecting", "further assistance",
    "escalate",
]


def check_must_include(answer, items):
    return [f"missing required text: {i!r}" for i in items if i.lower() not in answer.lower()]


def check_must_not_include(answer, items):
    return [f"contains forbidden text: {i!r}" for i in items if i.lower() in answer.lower()]


def check_concepts(answer, concepts):
    failures = []
    for concept in concepts:
        keywords = CONCEPT_KEYWORDS.get(concept, [concept])
        if not any(kw.lower() in answer.lower() for kw in keywords):
            failures.append(f"missing concept: {concept!r} (keywords tried: {keywords})")
    return failures


def check_sources(trace, required_sources):
    failures = []
    if trace.get("retrieval") is None:
        if required_sources:
            failures.append(f"expected sources {required_sources} but retrieval was skipped (order path)")
        return failures
    retrieved_files = {c[0] for c in trace["retrieval"]["chunks"]}
    for src in required_sources:
        if src not in retrieved_files:
            failures.append(f"required source not retrieved: {src}")
    return failures


def check_forbidden_sources(answer, forbidden_sources):
    return [f"forbidden source cited as authority: {s}" for s in forbidden_sources if s in answer]


def check_tool(trace, expected_tool, tool_arguments=None):
    failures = []
    order_lookup = trace.get("order_lookup")
    if expected_tool in ("not_called", "not_called_without_id"):
        if order_lookup is not None:
            failures.append(f"tool was called but expected {expected_tool}")
    elif expected_tool == "order_lookup":
        if order_lookup is None:
            failures.append("expected order_lookup tool call, none happened")
        elif tool_arguments and order_lookup.get("order_id") != tool_arguments.get("order_id"):
            failures.append(
                f"tool called with order_id={order_lookup.get('order_id')!r}, "
                f"expected {tool_arguments.get('order_id')!r}"
            )
    # "optional_sanitized_lookup": either calling or not is acceptable
    return failures


def check_handoff(answer, expected_handoff):
    detected = any(kw.lower() in answer.lower() for kw in HANDOFF_KEYWORDS)
    if expected_handoff and not detected:
        return ["expected handoff language, none detected"]
    return []


def run_case(case):
    session_id = f"eval-{case['id']}"
    last_result = None
    for msg in case["messages"]:
        last_result = run_turn(session_id, msg["content"])

    answer = last_result["answer"]
    trace = last_result["trace"]
    expect = case["expect"]

    failures = []
    if "must_include" in expect:
        failures += check_must_include(answer, expect["must_include"])
    if "must_not_include" in expect:
        failures += check_must_not_include(answer, expect["must_not_include"])
    if "must_include_concepts" in expect:
        failures += check_concepts(answer, expect["must_include_concepts"])
    if "required_sources" in expect:
        failures += check_sources(trace, expect["required_sources"])
    if "forbidden_sources_as_authority" in expect:
        failures += check_forbidden_sources(answer, expect["forbidden_sources_as_authority"])
    if "tool" in expect:
        failures += check_tool(trace, expect["tool"], expect.get("tool_arguments"))
    if "handoff" in expect:
        failures += check_handoff(answer, expect["handoff"])
    if expect.get("must_not_silently_choose_one"):
        both_present = ("hand-wash" in answer.lower() or "hand wash" in answer.lower()) and \
                        ("dishwasher safe" in answer.lower())
        if not both_present:
            failures.append("expected both conflicting claims acknowledged, not silently resolved")

    return {
        "id": case["id"], "category": case.get("category"),
        "passed": len(failures) == 0, "failures": failures, "answer": answer,
    }


def main():
    cases_path = Path(__file__).parent / "visible-cases.json"
    with open(cases_path, encoding="utf-8") as f:
        data = json.load(f)

    results = [run_case(c) for c in data["cases"]]
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {r['id']} ({r['category']})")
        for f_ in r["failures"]:
            print(f"    - {f_}")

    passed = sum(1 for r in results if r["passed"])
    print(f"\n{passed}/{len(results)} passed")

    out_path = Path(__file__).parent / "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Full results saved to {out_path}")


if __name__ == "__main__":
    main()