# Catalist Temporal RAG & Bi-Temporal Knowledge Graph Pipeline

An autonomous, bi-temporal Retrieval-Augmented Generation (RAG) system engineered over the **138 AMI Meeting Corpus transcripts** (`35` design teams, `676,245` words) to resolve multi-hop chronological queries with deterministic single-turn verbatim citations, synthesize point-in-time executive briefings (`TS3012`), and self-evaluate via **DeepEval** and **Pytest**.

---

## 1. Architectural Highlights

1. **Strict Model-vs-Code Boundary (`RapidFuzz` Deterministic Citation Gate)**:
   - The LLM is categorically prohibited from writing the final `quote` string in [`answers.json`](file:///d:/Catalist_Project/answers.json).
   - Every candidate citation is routed through [`locate_verbatim_quote()`](file:///d:/Catalist_Project/src/citation/verbatim_extractor.py#L140-L246) in [`src/citation/verbatim_extractor.py`](file:///d:/Catalist_Project/src/citation/verbatim_extractor.py), which executes disfluency-aware character-offset alignment (`rapidfuzz.fuzz.partial_ratio >= 95.0`) against the raw `.txt` files and slices the exact character range—preserving 100% of spoken disfluencies (`"um"`, `"uh"`) and AMI corpus notation (`L_C_D_`, `T_V_`).
2. **Bi-Temporal Knowledge Graph (`valid_from` & `invalid_at`)**:
   - Implemented in [`src/graph/bitemporal_graph.py`](file:///d:/Catalist_Project/src/graph/bitemporal_graph.py) following the Graphiti/Zep bi-temporal paradigm.
   - Tracks both **transaction time** (`created_at`, `deleted_at`) and **domain validity intervals** (`[valid_from, invalid_at)` where `a=1.0, b=2.0, c=3.0, d=4.0`), mutating superseded decision edges with `invalid_at = t_supersede` and linking evolution chains via explicit `SUPERSEDES` / `SUPERSEDED_BY` relationships (`1,191` nodes, `1,612` edges, `488` decisions: `352` active frontier, `136` superseded historical states, `109` evolution chains).
3. **Distributed Subagent Ingestion (`invoke_subagent`)**:
   - Engineered in [`src/ingestion/subagent_orchestrator.py`](file:///d:/Catalist_Project/src/ingestion/subagent_orchestrator.py) to partition the 35 teams across 5 parallel `ami-extractor` worker subagents and index `35,381` diarization-preserving chunks into Qdrant (`data/qdrant_db`).
4. **Automated DeepEval Self-Evaluation (`tests/test_agent.py`)**:
   - Evaluates 10 novel multi-hop chronological questions (`CQ01`–`CQ10` in [`src/evaluation/custom_benchmark.py`](file:///d:/Catalist_Project/src/evaluation/custom_benchmark.py)) using `FaithfulnessMetric` and `HallucinationMetric`, verifying both positive multi-hop recall (`CQ01`–`CQ07`) and 3 diagnostic failure modes (`CQ08`–`CQ10`), alongside full schema and substring verification of [`answers.json`](file:///d:/Catalist_Project/answers.json).

---

## 2. Repository Structure

```text
d:\Catalist_Project\
├── AGENTS.md                                      # Principal Data Engineer architectural doctrines
├── README.md                                      # Setup, architecture, and execution guide
├── WRITEUP.md                                     # 1-page technical write-up, scaling math & failure post-mortems
├── answers.json                                   # Final deliverable: 25 multi-hop answers + TS3012 team brief
├── questions.json                                 # Held-back 25 multi-hop assessment queries (Q01-Q25)
├── transcripts/                                   # 138 raw AMI Meeting Corpus transcripts (*.txt)
├── .agents/
│   └── skills/verbatim-citation/SKILL.md          # Deterministic RapidFuzz citation skill specification
├── src/
│   ├── models/schemas.py                          # Strict Pydantic v2 schemas for chunks, graph edges & answers
│   ├── ingestion/
│   │   ├── parser.py                              # Diarization & disfluency-preserving transcript parser
│   │   ├── qdrant_indexer.py                      # Hybrid embeddings + temporal_order Qdrant indexer
│   │   ├── subagent_orchestrator.py               # 5-shard parallel subagent ingestion coordinator
│   │   └── normalize_subagent_bundles.py          # Subagent extraction bundle validator & normalizer
│   ├── graph/
│   │   └── bitemporal_graph.py                    # Bi-Temporal Knowledge Graph builder & Neo4j Cypher exporter
│   ├── citation/
│   │   └── verbatim_extractor.py                  # Deterministic RapidFuzz single-turn quote extractor
│   ├── resolution/
│   │   └── query_engine.py                        # Multi-hop query resolver (Q01-Q25) & TS3012 brief generator
│   └── evaluation/
│       └── custom_benchmark.py                    # 10 custom multi-hop benchmark questions (CQ01-CQ10)
├── tests/
│   └── test_agent.py                              # Pytest + DeepEval suite (Faithfulness & Hallucination metrics)
└── data/
    ├── extracted/*.json                           # 35 validated TeamExtractionBundle JSON files
    ├── bitemporal_graph.json                      # Serialized Bi-Temporal Knowledge Graph (1,191 nodes, 1,612 edges)
    ├── bitemporal_graph.cypher                    # Executable Neo4j Cypher graph dump
    ├── evaluation_results.json                    # DeepEval telemetry & failure mode diagnostics
    └── qdrant_db/                                 # Local Qdrant vector store (35,381 indexed chunks)
```

---

## 3. Environment Setup

Requires **Python 3.11+** (tested on Python `3.13.1`). Install all required dependencies:

```powershell
python -m pip install qdrant-client rapidfuzz deepeval pydantic neo4j networkx pytest
```

*(Note: On Windows environments where `python` is not on the default system `PATH`, invoke via `& "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"`.)*

---

## 4. End-to-End Execution Pipeline

Each phase of the pipeline can be executed standalone or end-to-end from the repository root (`d:\Catalist_Project`):

### Phase 1: Transcript Parsing, Subagent Normalization & Qdrant Indexing
Parses all 138 transcripts (`676,245` words, `70,405` speaker turns), validates all 35 subagent extraction bundles in `data/extracted/*.json`, and indexes `35,381` overlapping window chunks + atomic decision points into `data/qdrant_db`:

```powershell
python -m src.ingestion.normalize_subagent_bundles
python -m src.ingestion.qdrant_indexer
```

### Phase 2: Bi-Temporal Knowledge Graph Construction
Constructs the Bi-Temporal Knowledge Graph from `data/extracted/*.json`, applies supersession state mutations (`valid_from`, `invalid_at`, `SUPERSEDES` edges), and exports both `data/bitemporal_graph.json` and `data/bitemporal_graph.cypher`:

```powershell
python -m src.graph.bitemporal_graph
```

### Phase 3: Deterministic Citation Boundary Audit (`RapidFuzz`)
Audits all extracted semantic quotes against the raw `.txt` files in `transcripts/` using `locate_verbatim_quote(threshold=95.0)`:

```powershell
python -m src.citation.verbatim_extractor
```

### Phase 4: Query Resolution (`Q01`–`Q25`) & Team `TS3012` Brief Generation
Executes hybrid Qdrant + Bi-Temporal Graph resolution across `questions.json`, enforces single-turn verbatim citation extraction (`>= 6` words, `>= 2` meetings per question), synthesizes the point-in-time brief for Team `TS3012`, and writes the validated [`answers.json`](file:///d:/Catalist_Project/answers.json):

```powershell
python -m src.resolution.query_engine
```

### Phase 5: Self-Evaluation via DeepEval & Pytest
Runs the 10-question custom multi-hop benchmark (`CQ01`–`CQ10`), evaluates `FaithfulnessMetric` and `HallucinationMetric` via DeepEval, verifies the 3 diagnostic failure modes (`CQ08`, `CQ09`, `CQ10`), and validates [`answers.json`](file:///d:/Catalist_Project/answers.json):

```powershell
python -m pytest tests/test_agent.py -v
```
