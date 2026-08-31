# 🏔️ Aster & Row Support Agent

A RAG-based customer support agent built for the **AI Agent Intern take-home assignment**.

The agent answers customer-support questions using the supplied knowledge base, performs deterministic order lookups when an order ID is provided, detects policy/source conflicts, resists untrusted or superseded policy content, maintains multi-turn conversation context, and hands off to human support when it cannot safely make a decision.

## Features

* **Knowledge-base RAG** for customer-support policy and product questions
* **Deterministic order lookup** using the supplied order data
* **Source-aware retrieval** with document metadata and policy authority
* **Superseded-policy protection**
* **Source-conflict detection** with LLM-based final judgment
* **Prompt-injection resistance**
* **Multi-turn conversation context**
* **Citations** back to retrieved knowledge-base documents
* **Human-support handoff** when the agent cannot safely complete a request
* **Per-turn trace logging** for observability and debugging
* **15-case visible evaluation suite**
* **5 additional regression cases**
* **Local embeddings** with no external embedding API dependency

---

# Architecture

```text
                         ┌──────────────────────┐
                         │      User Query      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │       agent.py       │
                         │  Routing + response  │
                         │      generation      │
                         └──────────┬───────────┘
                                    │
                       ┌────────────┴────────────┐
                       │                         │
                Order ID present?          No Order ID
                       │                         │
                       ▼                         ▼
              ┌─────────────────┐       ┌─────────────────┐
              │  order_tool.py  │       │   retriever.py  │
              │                 │       │                 │
              │ Safe field      │       │ Query embedding │
              │ allow-list      │       │       ↓         │
              │       ↓         │       │ Cosine search   │
              │ orders.json     │       │       ↓         │
              │       ↓         │       │ Top-K + floor   │
              │ Sanitized result│       │       ↓         │
              └────────┬────────┘       │ Precedence      │
                       │                │ filtering       │
                       │                │       ↓         │
                       │                │ Conflict        │
                       │                │ candidates      │
                       │                └────────┬────────┘
                       │                         │
                       └────────────┬────────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │        Gemini       │
                         │                      │
                         │ Final answer         │
                         │ + citations          │
                         │ + conflict judgment  │
                         │ + handoff signal     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      User Answer     │
                         └──────────────────────┘
```

### Knowledge-base ingestion

```text
knowledge-base/*.md
        │
        ▼
   chunker.py
        │
        ├── separates YAML front matter
        ├── splits content by ## headings
        └── attaches metadata to each chunk
        │
        ▼
   indexer.py
        │
        └── all-MiniLM-L6-v2
        │
        ▼
derived/embeddings.npy
derived/chunks.jsonl
```

---

# Request Flow

## 1. Query Routing

`agent.py` first determines whether the message contains an order ID.

* **Order ID present:** use the deterministic order lookup path.
* **No order ID:** use the RAG retrieval path.

The two paths are mutually exclusive for a single message.

This is intentionally simple and transparent. A known limitation is that a message containing both an order lookup and an unrelated policy question currently follows the order-tool route rather than splitting the request into two independent tasks.

## 2. Retrieval

For policy and product questions:

1. The user query is embedded using `all-MiniLM-L6-v2`.
2. Cosine similarity is calculated against the stored chunk embeddings.
3. The top 10 candidates are selected, subject to a similarity floor.
4. Policy-authority filtering removes non-authoritative policy chunks before they reach the LLM.
5. Potential conflict candidates are identified using document metadata and lightweight text-pattern signals.
6. The final conflict decision is made in `agent.py` by Gemini reading the actual retrieved chunk text.

## 3. Order Lookup

Order queries use `order_tool.py` rather than RAG.

The tool:

* Validates the requested order ID.
* Normalizes harmless differences such as lowercase IDs and surrounding whitespace.
* Uses an allow-list of safe fields based on `orders-data-dictionary.md`.
* Reads the required information from `data/orders.json`.
* Returns only the sanitized lookup result to the LLM.

The full order record and raw order data are never inserted into the LLM prompt.

## 4. Generation

Gemini receives the relevant retrieved content or sanitized order result together with system instructions.

The system prompt controls:

* Grounding in retrieved information
* Citation requirements
* Policy precedence
* Superseded-document handling
* Conflict handling
* Human-support handoff
* Multi-turn behavior
* Prompt-injection resistance

The generation configuration uses `temperature=0` to reduce run-to-run phrasing variation and improve reproducibility of the evaluation suite.

---

# Why Conflict Judgment Lives in `agent.py`

An early implementation attempted to determine whether two chunks contradicted each other using embedding similarity between the chunks.

This approach was tested against the actual knowledge base and produced the wrong result.

The genuine Breeze Tumbler conflict involved:

* `11-product-care.md`: the stainless-steel body should be **hand-washed**
* `12-breeze-tumbler-product-card.md`: **all components are dishwasher safe**

The embedding-based approach incorrectly ranked the conflict relative to a genuine agreement between documents containing the same final-sale policy.

This demonstrated that sentence embeddings capture **topical/semantic similarity**, not logical agreement or contradiction.

The current implementation therefore uses:

* Metadata and lightweight text-pattern logic in `retriever.py` to identify **candidate conflicts**
* Gemini in `agent.py` to make the **final conflict judgment** from the actual retrieved source text

This keeps retrieval deterministic while allowing the LLM to perform the reading-comprehension part of conflict resolution.

---

# Knowledge Base Precedence

The system treats policy authority explicitly.

Official/current policy content is authoritative.

Superseded or migration content is not treated as an alternative policy simply because it was retrieved.

For example, if a migration note says that customers receive 60 days but the current official policy says:

* Standard customers: **30 calendar days**
* TrailPlus members: **45 calendar days**

the agent uses the current policy rather than presenting the migration note as a valid fallback.

Non-authoritative policy content is filtered before generation.

---

# Multi-Turn Conversations

Conversation context is maintained per `session_id`.

This allows follow-up questions such as:

```text
User: Do you ship internationally?

Agent: Yes, currently to Canada...

User: What about Canada, and how long does it take?

Agent: Canadian orders generally arrive within 5–9 business
days after dispatch...
```

The agent preserves materially relevant information from the earlier turn when continuing the same topic, rather than treating the follow-up as an unrelated question.

Session context is scoped to the current conversation so unrelated details are not carried indefinitely between sessions.

---

# Order Lookup

Order lookup is intentionally implemented as a deterministic Python tool rather than an LLM-generated lookup.

Example:

```text
User: Where is my order ORD-1007?

Agent: Your order ORD-1007 has a status: shipped.

      Carrier: UPS
      Tracking Number: 1ZAR100700000007
      Estimated Delivery: August 22, 2026
```

This separates:

* **Deterministic data retrieval** → Python
* **Natural-language response generation** → Gemini

This reduces the risk of the model inventing order information.

The tool does not expose customer email addresses, delivery addresses, internal notes, risk scores, or other internal-only fields.

---

# Safety and Prompt Injection

The system treats user messages, retrieved documents, and tool results as untrusted data.

The system does not rely exclusively on the LLM to ignore untrusted policy content.

Instead, policy authority is enforced before generation.

In particular:

```text
policy_authority != "official"
```

chunks are excluded from the LLM context by the retrieval pipeline when they are not authoritative for the requested policy.

The agent also refuses requests to reveal:

* System prompts
* Hidden instructions
* Secrets or credentials
* Internal-only order information
* Other information that should not be exposed to customers

The agent uses the supplied company content rather than general model knowledge for company-specific questions and abstains when the supplied information is insufficient.

---

# Evaluation

The project includes a deterministic evaluation suite covering all supplied visible cases and five additional original regression cases.

## Evaluation Coverage

| Category               |  Cases |    Result |
| ---------------------- | -----: | --------: |
| Retrieval              |      2 |       2/2 |
| Multi-source grounding |      1 |       1/1 |
| Conversation           |      1 |       1/1 |
| Groundedness           |      2 |       2/2 |
| Tool use               |      2 |       2/2 |
| Tool reliability       |      3 |       3/3 |
| Privacy                |      1 |       1/1 |
| Prompt security        |      1 |       1/1 |
| Abstention             |      1 |       1/1 |
| Source conflict        |      1 |       1/1 |
| **Visible cases**      | **15** | **15/15** |

The additional regression cases are defined in:

```text
evaluation/original-cases.json
```

These cases extend coverage beyond the supplied visible prompts and serve as regression tests for behaviors discovered during development.

## Baseline vs Final

**Baseline: 10/15**

**Final: 15/15 visible cases passed**

The main improvements between the baseline and final implementation were:

* Stronger policy-authority handling
* Improved retrieval coverage
* Safer superseded-policy behavior
* Improved conflict handling
* More reliable order-tool behavior
* Stronger prompt-injection handling
* Regression coverage for discovered failures

## Evaluation Command

Run the complete evaluation suite with:

```powershell
python evaluation\run_evals.py
```

Results are written to:

```text
evaluation/eval_results.json
```

The evaluation harness was not modified to loosen keyword matching or artificially increase the pass rate.

The final **15/15 visible-case result** was achieved with the agent implementation and `temperature=0` generation configuration.

---

# Bug Diary

The project includes a detailed development bug diary at:

```text
evaluation/bug-diary.md
```

The following are the major failures identified during development.

## 1. Superseded Policy Presented as a Valid Fallback

**Reproduction:**

Ask the agent about the return window using wording that could retrieve both current and legacy return policies.

**Failure:**

The agent initially treated the superseded policy as an alternative policy for customers not covered by the current policy.

**Root Cause:**

The initial implementation relied too heavily on semantic retrieval and did not sufficiently enforce document authority before generation.

**Fix:**

Added explicit policy precedence and supersession rules so superseded documents cannot override current official policy.

**Regression Test:**

A return-policy evaluation case verifies that the current policy is selected and the legacy policy is not presented as a valid alternative.

---

## 2. Embedding-Based Conflict Detection Produced Incorrect Reasoning

**Reproduction:**

Compare the Breeze Tumbler care instructions with the product card and attempt to identify contradictions using chunk-to-chunk embedding similarity.

**Failure:**

The similarity heuristic incorrectly ranked the genuine conflict relative to documents that actually agreed on the same final-sale policy.

**Root Cause:**

Embedding similarity measures semantic/topic similarity, not logical contradiction.

**Fix:**

Removed embedding similarity as the final conflict judgment mechanism.

`retriever.py` now identifies candidate conflicts using metadata and lightweight text signals, while Gemini makes the final judgment using the actual retrieved source text.

**Regression Test:**

The source-conflict evaluation case verifies that the Breeze Tumbler contradiction is surfaced instead of silently selecting one source.

---

## 3. Retrieval Coverage Gap

**Reproduction:**

Use an adversarially phrased policy query where the correct current policy initially ranked outside the retrieved candidate set.

**Failure:**

With `TOP_K=5`, the correct current policy was not consistently included in the retrieved context.

**Root Cause:**

The candidate retrieval set was too small for some paraphrased/adversarial queries.

**Fix:**

Increased retrieval coverage to `TOP_K=10`, while retaining the policy-authority filter afterward.

This improves recall without allowing non-authoritative policy content to bypass the precedence filter.

**Regression Test:**

The retrieval evaluation suite includes paraphrased policy queries that verify the correct authoritative source remains retrievable.

---

## 4. Order Data Could Be Overexposed

**Reproduction:**

Test an order query while inspecting the data passed from the order tool to the generation step.

**Failure:**

An early implementation exposed more raw order fields than were necessary for the customer-facing answer.

**Root Cause:**

The tool returned the order record too broadly instead of enforcing an explicit customer-safe field allow-list.

**Fix:**

Added an allow-list based on `orders-data-dictionary.md` and return only sanitized customer-safe fields.

**Regression Test:**

Privacy evaluation cases verify that customer email, address, internal notes, risk scores, and other internal-only fields are not returned.

---

# Project Structure

```text
ai-agent-intern-test/

│
├── knowledge-base/
│   ├── 01-returns-policy-current.md
│   ├── 02-returns-policy-legacy.md
│   ├── 03-final-sale-and-promotions.md
│   ├── ...
│   └── 14-internal-content-migration-notes.md
│
├── data/
│   ├── orders.json
│   └── orders-data-dictionary.md
│
├── src/
│   ├── agent.py
│   ├── chunker.py
│   ├── indexer.py
│   ├── retriever.py
│   └── order_tool.py
│
├── evaluation/
│   ├── visible-cases.json
│   ├── original-cases.json
│   ├── run_evals.py
│   ├── bug-diary.md
│   └── eval_results.json
│
├── derived/
│   ├── embeddings.npy
│   └── chunks.jsonl
│
├── logs/
│   └── trace_log.jsonl
│
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

# Setup

## 1. Clone the Repository

```powershell
git clone <YOUR-REPOSITORY-URL>

cd ai-agent-intern-test
```

Replace `<YOUR-REPOSITORY-URL>` with the GitHub repository URL.

## 2. Create a Virtual Environment

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

## 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

## 4. Configure the Gemini API Key

Copy the environment template:

```powershell
copy .env.example .env
```

Add the Gemini API key:

```text
GOOGLE_API_KEY=your_api_key_here
```

No embedding API key is required because embeddings run locally.

**Do not commit `.env` or any real credentials to the repository.**

## 5. Build the Retrieval Index

Run this once after cloning, and again whenever the knowledge base changes:

```powershell
python src\indexer.py
```

This creates:

```text
derived/embeddings.npy
derived/chunks.jsonl
```

## 6. Run the CLI Agent

```powershell
python src\agent.py
```

## 7. Run the Web Interface

```powershell
streamlit run app.py
```

## 8. Run Evaluations

```powershell
python evaluation\run_evals.py
```

---

# Technology Choices

## Generation

**Gemini** through the `google-genai` SDK.

`temperature=0` is used to reduce output variability and improve reproducibility of the deterministic evaluation suite.

## Embeddings

**Sentence Transformers — `all-MiniLM-L6-v2`**

Embeddings run locally, avoiding dependency on a hosted embedding API.

The corpus is small enough that local embeddings plus brute-force NumPy cosine similarity are sufficient.

## Retrieval

No LangChain, LlamaIndex, Chroma, or external vector database is used.

For the supplied corpus size, NumPy cosine similarity provides a small, transparent implementation of vector retrieval without introducing unnecessary infrastructure.

## Storage

The retrieval index is stored as:

```text
derived/embeddings.npy
derived/chunks.jsonl
```

The supplied knowledge base and order data are treated as source data and are not modified.

---

# Observability

Each agent turn is recorded in:

```text
logs/trace_log.jsonl
```

The trace makes it possible to inspect:

* Current user message
* Relevant conversation history
* Retrieved passages
* Document metadata
* Retrieval scores
* Candidate conflicts
* Tool calls
* Sanitized tool results
* Final response
* Errors, fallbacks, and handoffs

Sensitive customer fields and secrets are not intentionally logged.

Debug/trace information is intended for development and evaluation rather than customer-facing output.

---

# Known Limitations

This implementation is intentionally scoped to the supplied corpus and assignment requirements.

## Evaluation Grading

The supplied evaluation harness relies on literal/concept keyword matching rather than full semantic grading.

For a production system, I would combine deterministic assertions with a more robust semantic or rubric-based evaluator.

## Conflict Detection

Candidate conflicts are narrowed using metadata and text-pattern signals.

The final judgment is made by the LLM rather than a dedicated NLI/contradiction model.

This works well for the supplied corpus but may need a more specialized approach at larger scale.

## Retrieval Candidate Noise

Increasing `TOP_K` from 5 to 10 improves retrieval coverage but can result in a noisier candidate set.

The current system relies on the policy-authority filter and final LLM judgment to handle this trade-off.

## Combined Order + Policy Questions

A message containing both an order lookup and an unrelated policy question currently follows the order-tool route.

For example:

```text
Where is ORD-1007 and is the Breeze Tumbler dishwasher safe?
```

is not split into two independent requests.

This is documented rather than silently pretending both questions were answered.

## API Quota

The Gemini free tier has request limits.

A production deployment would require appropriate API capacity or a locally hosted generation model.

## Schema Synchronization

The order-tool field allow-list was built and tested against the supplied `orders-data-dictionary.md` and supplied orders.

It is not automatically regenerated if the source schema changes.

## Conversation Storage

Conversation context is maintained in application memory and is scoped to a session.

A production implementation would require a durable session store and explicit retention policies.

---

# AI Coding Tools Used

AI coding assistants were used as pair-programming collaborators during development.

## Claude

Claude was used for:

* Architecture discussion
* Code generation
* Debugging
* Directed adversarial testing
* Evaluation analysis
* Documentation and README review

AI-generated suggestions were treated as proposals rather than trusted implementations and were verified against the assignment requirements and actual test results.

## Example of an Incorrect AI-Generated Suggestion

An AI-generated approach suggested using embedding similarity between retrieved chunks to determine whether two documents agreed or contradicted each other.

Testing this against the actual knowledge base showed that the approach was unreliable: semantic similarity measured topical similarity rather than logical agreement or contradiction.

The approach was therefore removed.

The final implementation uses deterministic retrieval and candidate-conflict detection, followed by LLM-based reasoning over the actual source text.

---

# Demo

The demo is a **2–4 minute walkthrough** of the completed agent.

It demonstrates:

1. A knowledge-base question with citations
2. A deterministic order lookup
3. A multi-turn conversation
4. A case where the agent refuses to guess or recommends human support
5. Superseded-policy and source-conflict handling
6. The evaluation suite achieving **15/15**

## Demo Video

**[▶ Watch the Aster & Row Support Agent Demo](https://drive.google.com/file/d/1lfSSt-3gy7wT6tkTWIqihRsT_2ZwCkIs/view?usp=drive_link)**



---

# Summary

The implementation deliberately favors a small, transparent architecture over unnecessary framework complexity:

```text
Local embeddings
      +
Simple vector retrieval
      +
Explicit policy precedence
      +
Deterministic order tool
      +
LLM response generation
      +
Conflict/handoff handling
      +
Evaluation + trace logging
```

The result is a compact customer-support agent that is grounded in the supplied data, resistant to known policy pitfalls, observable during execution, and validated against the assignment's visible evaluation suite.

**Baseline:** 10/15

**Final:** 15/15 visible cases passed
