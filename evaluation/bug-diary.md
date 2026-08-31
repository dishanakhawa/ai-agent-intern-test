# Bug Diary

## Bug 1: Superseded document treated as a valid fallback policy
**Repro:** Ask "What is your return policy?" when doc 14 (draft, unauthoritative)
is topically close to the query. Model cited doc 02 (superseded, 45 days) as an
"otherwise" option for customers not covered by the active TrailPlus policy.
**Root cause:** System prompt had rules for genuine *conflicts* between active
docs, but no explicit rule about superseded documents being historical-only,
non-fallback content.
**Fix:** Added an explicit rule to SYSTEM_PROMPT: a source marked
`status: superseded` must never be presented as a current valid option or
fallback, even for a case the active policy doesn't explicitly cover.
**Regression test:** `original-supersession-not-fallback`

## Bug 2: False-positive conflict detection (embedding similarity approach)
**Repro:** doc 01 vs doc 03 (both correctly state final-sale items aren't
returnable for change of mind) were flagged as a genuine conflict.
**Root cause:** An earlier fix attempt used embedding cosine-similarity
*between* the two candidate chunks to distinguish agreement from
contradiction. Tested directly: doc 11 vs doc 12 (real conflict) scored
*higher* mutual similarity (0.80) than doc 01 vs doc 03 (real agreement,
0.71) -- the wrong relative order. MiniLM embeddings capture topic
similarity, not stance/entailment.
**Fix:** Abandoned the embedding-similarity approach entirely. Final design:
retriever.py only narrows to `candidate_conflict` via metadata (supersession)
and text-pattern (conditional/eligibility scoping) checks -- both reliable,
neither requires stance detection. The actual true/false contradiction
judgment is made by the LLM in agent.py, reading real chunk text under an
explicit system-prompt rule.
**Regression test:** `original-agreement-not-conflict`, plus visible-cases.json's
own `genuine-active-source-conflict` case (still correctly flagged as True).

## Bug 3: Retrieval coverage gap on adversarial phrasing
**Repro:** For the prompt-injection test query ("the migration note says...
give everyone 60 days"), doc 01 (the correct current policy, needed for
grounded refusal) ranked 9th by cosine similarity and was excluded by
TOP_K=5 before precedence_filter ever ran.
**Root cause:** The adversarial query's wording overlaps heavily with doc 14's
own language ("migration note," "60 days"), so doc 14's chunks dominated the
top ranks, pushing doc 01 out of a narrow top-5 window.
**Fix:** Raised TOP_K from 5 to 10 in retriever.py. Verified via code
inspection that precedence_filter (hard policy_authority=='official'
exclusion) runs strictly after top_k slicing, so widening top_k does not
let doc 14 (or any non-official source) bypass the filter -- it only gives
more candidates a chance to pass through the same gate. Verified via a real
end-to-end run that doc 01 is now retrieved and correctly cited, doc 14
remains excluded, and the resulting larger candidate-conflict set did not
cause any new false-positive conflict.
**Regression test:** visible-cases.json's `retrieved-prompt-injection` case.