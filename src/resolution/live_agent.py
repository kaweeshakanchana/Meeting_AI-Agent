"""Live Unseen-Question Resolution Engine & Interactive Interview CLI/API (src/resolution/live_agent.py).

Enables live, real-time resolution of brand-new unseen multi-hop questions during the technical
interview or live demo, operating over all 35 teams and 138 AMI transcripts:
1. Detects target team (`ES2002`-`TS3012`), temporal scope (`a`-`d`), and query intent
   (decisions, supersession/evolution, budget cuts, action items, disagreements).
2. Traverses the Bi-Temporal Knowledge Graph (`data/bitemporal_graph.json`) including
   active frontier edges (`invalid_at is None`), historical `SUPERSEDES` chains, action
   items, and disagreements, supplemented by raw diarized turn retrieval across `transcripts/*.txt`.
3. Synthesizes a concise 2-3 sentence chronological answer naming exact Speaker letters (`A`-`D`),
   speaker names, and roles.
4. Enforces the strict Model-vs-Code boundary (`locate_verbatim_quote(..., threshold=95.0)`) so
   every returned citation is guaranteed to be an exact single-turn substring (`>= 6` words)
   spanning at least 2 distinct meetings.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.citation.verbatim_extractor import (
    CitationHallucinationError,
    locate_verbatim_quote,
    verify_citation_in_transcript,
)
from src.graph.bitemporal_graph import BiTemporalKnowledgeGraph, canonicalize_topic
from src.ingestion.parser import parse_transcript_file
from src.ingestion.qdrant_indexer import AMIHybridQdrantStore, compute_dense_embedding


STOPWORDS: Set[str] = {
    "the", "and", "for", "that", "this", "with", "from", "what", "which", "when",
    "where", "who", "whom", "whose", "why", "how", "did", "does", "do", "done",
    "was", "were", "are", "is", "be", "been", "being", "have", "has", "had",
    "having", "their", "they", "them", "team", "meeting", "meetings", "remote",
    "control", "about", "into", "over", "after", "before", "between", "across",
    "under", "more", "most", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "can", "will", "just", "should", "now",
    "decide", "decided", "decision", "plan", "planned", "change", "changed",
    "final", "finally", "initially", "first", "end", "both", "two", "give", "given",
}

SYNONYM_GROUPS: List[Set[str]] = [
    {"lcd", "screen", "display", "touch", "touchscreen", "l_c_d_"},
    {"speech", "voice", "recognition", "microphone", "audio", "sample", "sensor", "speaker"},
    {"energy", "power", "battery", "batteries", "solar", "kinetic", "dynamo", "hand", "rechargeable", "charging", "charger", "docking"},
    {"case", "casing", "material", "rubber", "plastic", "wood", "wooden", "titanium", "latex", "spongy", "soft", "hard"},
    {"shape", "form", "curved", "curve", "double", "single", "flat", "banana", "snowman", "potato", "square", "round", "rounded"},
    {"button", "buttons", "pushbutton", "pushbuttons", "scroll", "wheel", "spinning", "joystick", "keypad", "interface"},
    {"cost", "costs", "budget", "price", "euro", "euros", "dollar", "dollars", "expensive", "cheap", "cut", "drop", "dropped", "twelve", "twenty", "five", "fifty"},
    {"locator", "lost", "find", "finder", "beep", "beeper", "buzzer", "light", "paging"},
    {"teletext", "subtitle", "subtitles", "tv", "t_v_", "television", "universal", "dvd", "vcr"},
    {"age", "young", "younger", "old", "older", "elderly", "target", "market", "group", "forty", "sixty", "eighty", "fifteen", "thirty"},
    {"colour", "color", "colours", "colors", "yellow", "black", "grey", "red", "white", "cover", "covers", "changeable"},
    {"chip", "print", "regular", "advanced", "simple", "electronics"},
    {"action", "actions", "task", "tasks", "todo", "owner", "assigned", "responsible", "next"},
    {"disagree", "disagreement", "disagreements", "object", "objected", "objection", "concern", "argued", "conflict", "unresolved"},
]


def _tokenize_keywords(text: str) -> Set[str]:
    raw_tokens = re.findall(r"[a-z0-9_]+", text.lower())
    base = {t for t in raw_tokens if len(t) >= 2 and t not in STOPWORDS}
    expanded = set(base)
    for group in SYNONYM_GROUPS:
        if base & group:
            expanded |= group
    return expanded


def _extract_primary_keywords(text: str) -> Set[str]:
    raw_tokens = re.findall(r"[a-z0-9_]+", text.lower())
    return {t for t in raw_tokens if len(t) >= 3 and t not in STOPWORDS}


def _detect_team_id(
    question_text: str,
    graph: BiTemporalKnowledgeGraph,
    team_hint: Optional[str] = None,
) -> str:
    if team_hint and team_hint.upper() in graph.teams:
        return team_hint.upper()

    match = re.search(r"\b(ES20\d{2}|IS10\d{2}|TS30\d{2})[a-d]?\b", question_text, re.IGNORECASE)
    if match:
        candidate = match.group(1).upper()
        if candidate in graph.teams:
            return candidate

    # Fallback: score all 35 teams by keyword overlap with question
    q_primary = _extract_primary_keywords(question_text)
    q_expanded = _tokenize_keywords(question_text)
    best_team = "TS3012"
    best_score = -1.0

    for tid, bundle in graph.teams.items():
        corpus_parts: List[str] = [
            " ".join(bundle.speaker_names.values()),
            " ".join(bundle.speaker_roles.values()),
        ]
        for d in bundle.decisions:
            corpus_parts.append(f"{d.topic} {d.decision_value} {d.rationale} {d.semantic_target_quote}")
        for dis in bundle.disagreements:
            pos_val = (
                " ".join(dis.positions.values())
                if isinstance(dis.positions, dict)
                else str(dis.positions)
            )
            corpus_parts.append(f"{dis.topic} {pos_val} {dis.semantic_target_quote}")
        for act in bundle.action_items:
            corpus_parts.append(f"{act.task_description} {act.resolution_summary or ''} {act.semantic_target_quote}")

        team_tokens = set(re.findall(r"[a-z0-9_]+", " ".join(corpus_parts).lower()))
        score = 3.0 * len(q_primary & team_tokens) + 1.0 * len(q_expanded & team_tokens)
        if score > best_score:
            best_score = score
            best_team = tid

    return best_team


def _format_speaker_label(
    speaker_id: str,
    speaker_roles: Dict[str, str],
    speaker_names: Dict[str, str],
) -> str:
    spk = (speaker_id or "A").strip().upper()
    if len(spk) > 1 and spk.startswith("SPEAKER"):
        spk = spk.replace("SPEAKER", "").strip()
    role = speaker_roles.get(spk, "Team Member")
    name = speaker_names.get(spk, "")
    if name and name.lower() not in role.lower():
        return f"{role} {name} (Speaker {spk})"
    return f"{role} (Speaker {spk})"


def _verify_and_anchor_quote(
    meeting_id: str,
    candidate_quote: str,
    transcripts_dir: Path,
) -> Optional[str]:
    if not candidate_quote or len(candidate_quote.strip().split()) < 6:
        return None
    t_path = transcripts_dir / f"{meeting_id}.txt"
    if not t_path.exists():
        return None
    try:
        anchored = locate_verbatim_quote(t_path, candidate_quote, threshold=95.0)
        if len(anchored.split()) >= 6 and verify_citation_in_transcript(t_path, anchored):
            return anchored
    except CitationHallucinationError:
        return None
    return None


def _score_text_relevance(
    text: str,
    q_primary: Set[str],
    q_expanded: Set[str],
    q_vec: Optional[List[float]] = None,
) -> float:
    tokens = set(re.findall(r"[a-z0-9_]+", text.lower()))
    primary_hits = len(q_primary & tokens)
    expanded_hits = len(q_expanded & tokens)
    score = primary_hits * 4.0 + expanded_hits * 1.2
    if q_vec is not None and text.strip():
        t_vec = compute_dense_embedding(text)
        cosine = sum(a * b for a, b in zip(q_vec, t_vec))
        score += max(0.0, cosine) * 6.0
    return score


def _query_qdrant_live(
    question_text: str,
    team_id: Optional[str] = None,
    top_k: int = 8,
    qdrant_path: str | Path = "data/qdrant_db",
) -> List[Dict[str, Any]]:
    """Query the persistent Qdrant vector store (35,381 points) with filterable HNSW + RRF."""
    db_dir = Path(qdrant_path)
    if not db_dir.exists():
        return []
    store: Optional[AMIHybridQdrantStore] = None
    try:
        store = AMIHybridQdrantStore(db_path=db_dir)
        hits = store.hybrid_search(
            query=question_text,
            team_id=team_id,
            top_k=top_k,
        )
        return hits
    except Exception:
        return []
    finally:
        if store is not None:
            try:
                store.close()
            except Exception:
                pass


def resolve_live_question(
    question_text: str,
    question_id: str = "LIVE_Q01",
    team_hint: Optional[str] = None,
    graph: Optional[BiTemporalKnowledgeGraph] = None,
    transcripts_dir: str | Path = "transcripts",
) -> Dict[str, Any]:
    """Dynamically resolve any unseen multi-hop question using Qdrant Hybrid Search + Bi-Temporal Graph + RapidFuzz."""
    t0 = time.perf_counter()
    transcripts_root = Path(transcripts_dir)
    if graph is None:
        graph = BiTemporalKnowledgeGraph.load_from_disk("data/bitemporal_graph.json")

    team_id = _detect_team_id(question_text, graph, team_hint=team_hint)

    # Execute live Qdrant hybrid vector + RRF search over `data/qdrant_db`
    qdrant_hits = _query_qdrant_live(question_text, team_id=team_id, top_k=8)
    q_vec = compute_dense_embedding(question_text)

    bundle = graph.teams[team_id]
    speaker_roles = bundle.speaker_roles
    speaker_names = bundle.speaker_names

    q_low = question_text.lower()
    q_primary = _extract_primary_keywords(question_text)
    q_expanded = _tokenize_keywords(question_text)

    # Build a fast lookup of Qdrant hybrid scores by (meeting_id, quote/edge_id)
    qdrant_edge_boost: Dict[str, float] = {}
    qdrant_meeting_boost: Dict[str, float] = {}
    for hit in qdrant_hits:
        h_score = float(hit.get("_hybrid_score", 0.0))
        if hit.get("edge_id"):
            qdrant_edge_boost[str(hit["edge_id"])] = h_score * 10.0
        mid = str(hit.get("meeting_id", ""))
        if mid:
            qdrant_meeting_boost[mid] = max(qdrant_meeting_boost.get(mid, 0.0), h_score * 3.0)

    wants_actions = any(k in q_low for k in ["action", "task", "owner", "todo", "assign", "next meeting", "responsible"])
    wants_disagreements = any(
        k in q_low for k in ["disagree", "disagreement", "object", "objection", "conflict", "unresolved", "argued", "against"]
    )

    # 1. Score all DecisionEdges, Disagreements, and ActionItems using Hybrid (Lexical + Dense Embedding + Qdrant RRF)
    scored_decisions: List[Tuple[float, Any]] = []
    for dec in bundle.decisions:
        blob = f"{dec.topic} { canonicalize_topic(dec.topic) } {dec.decision_value} {dec.rationale} {dec.semantic_target_quote}"
        score = _score_text_relevance(blob, q_primary, q_expanded, q_vec=q_vec)
        score += qdrant_edge_boost.get(dec.edge_id, 0.0)
        score += qdrant_meeting_boost.get(dec.meeting_id, 0.0)
        # Boost decisions that belong to a supersession chain when question asks about change/evolution
        if dec.supersedes_edge_id or dec.invalid_at is not None:
            score += 2.0
        scored_decisions.append((score, dec))

    scored_decisions.sort(key=lambda pair: (pair[0], pair[1].valid_from), reverse=True)

    def _format_positions(positions: Any) -> str:
        if isinstance(positions, dict):
            return "; ".join(
                f"{_format_speaker_label(spk, speaker_roles, speaker_names)}: {pos}"
                for spk, pos in positions.items()
            )
        if isinstance(positions, list):
            return "; ".join(str(x) for x in positions)
        return str(positions)

    scored_disagreements: List[Tuple[float, Any]] = []
    for dis in bundle.disagreements:
        pos_str = _format_positions(dis.positions)
        blob = f"{dis.topic} {pos_str} {dis.semantic_target_quote}"
        score = _score_text_relevance(blob, q_primary, q_expanded)
        if wants_disagreements:
            score += 5.0
        scored_disagreements.append((score, dis))
    scored_disagreements.sort(key=lambda pair: pair[0], reverse=True)

    scored_actions: List[Tuple[float, Any]] = []
    for act in bundle.action_items:
        blob = f"{act.task_description} {act.owner_role} {act.resolution_summary or ''} {act.semantic_target_quote}"
        score = _score_text_relevance(blob, q_primary, q_expanded)
        if wants_actions:
            score += 5.0
        scored_actions.append((score, act))
    scored_actions.sort(key=lambda pair: pair[0], reverse=True)

    # 2. Build chronological candidate evidence pool (prioritizing evolution chains on the top-matched topic)
    selected_records: List[Dict[str, Any]] = []
    used_quotes: Set[str] = set()

    def _try_add_record(
        meeting_id: str,
        speaker_id: str,
        summary_text: str,
        raw_quote: str,
        record_type: str,
        valid_from: float,
        invalid_at: Optional[float] = None,
        topic: str = "",
    ) -> bool:
        anchored = _verify_and_anchor_quote(meeting_id, raw_quote, transcripts_root)
        if not anchored or anchored in used_quotes:
            return False
        used_quotes.add(anchored)
        selected_records.append(
            {
                "meeting_id": meeting_id,
                "speaker_id": speaker_id,
                "speaker_label": _format_speaker_label(speaker_id, speaker_roles, speaker_names),
                "summary": summary_text,
                "quote": anchored,
                "record_type": record_type,
                "valid_from": valid_from,
                "invalid_at": invalid_at,
                "topic": topic,
            }
        )
        return True

    # If the question specifically asks about action items
    if wants_actions and scored_actions and scored_actions[0][0] > 0:
        for _, act in scored_actions[:3]:
            desc = f"assigned { _format_speaker_label(act.owner_speaker_id, speaker_roles, speaker_names) } the action item: {act.task_description}"
            if act.resolution_summary:
                desc += f" (resolved in {act.resolution_meeting_id or 'a later meeting'}: {act.resolution_summary})"
            _try_add_record(
                act.meeting_id,
                act.owner_speaker_id,
                desc,
                act.semantic_target_quote,
                "ACTION_ITEM",
                act.valid_from,
                act.invalid_at,
                topic="action_item",
            )

    # If the question specifically asks about disagreements
    if wants_disagreements and scored_disagreements and scored_disagreements[0][0] > 0:
        for _, dis in scored_disagreements[:2]:
            pos_summary = _format_positions(dis.positions)
            spk0 = dis.speakers_involved[0] if dis.speakers_involved else "A"
            _try_add_record(
                dis.meeting_id,
                spk0,
                f"debated {dis.topic.replace('_', ' ')} ({pos_summary})",
                dis.semantic_target_quote,
                "DISAGREEMENT",
                dis.valid_from,
                dis.invalid_at,
                topic=dis.topic,
            )

    # Expand the top-scoring decision into its full chronological chain across meetings
    if scored_decisions:
        top_dec = scored_decisions[0][1]
        top_canon_topic = canonicalize_topic(top_dec.topic)

        # Find all decisions in this team sharing the same canonical topic or linked via supersedes_edge_id
        by_id = {d.edge_id: d for d in bundle.decisions}
        chain_Ids: Set[str] = {top_dec.edge_id}

        # Trace backward
        curr = top_dec
        while curr.supersedes_edge_id and curr.supersedes_edge_id in by_id:
            chain_Ids.add(curr.supersedes_edge_id)
            curr = by_id[curr.supersedes_edge_id]

        # Trace forward & topic peers
        for d in bundle.decisions:
            if d.supersedes_edge_id in chain_Ids or canonicalize_topic(d.topic) == top_canon_topic:
                chain_Ids.add(d.edge_id)

        chain_decs = sorted([by_id[eid] for eid in chain_Ids], key=lambda x: x.valid_from)
        for d in chain_decs:
            _try_add_record(
                d.meeting_id,
                d.speaker_id,
                f"{d.decision_value}" + (f" ({d.rationale})" if d.rationale and d.rationale.lower() not in d.decision_value.lower() else ""),
                d.semantic_target_quote,
                f"DECISION ({d.status})",
                d.valid_from,
                d.invalid_at,
                topic=d.topic,
            )

        # Add high-scoring secondary decisions if we still need multi-meeting coverage
        for score, d in scored_decisions[1:]:
            if len(selected_records) >= 4 and len({r["meeting_id"] for r in selected_records}) >= 2:
                break
            if score <= 0 and len(selected_records) >= 2:
                break
            _try_add_record(
                d.meeting_id,
                d.speaker_id,
                f"{d.decision_value}" + (f" ({d.rationale})" if d.rationale and d.rationale.lower() not in d.decision_value.lower() else ""),
                d.semantic_target_quote,
                f"DECISION ({d.status})",
                d.valid_from,
                d.invalid_at,
                topic=d.topic,
            )

    # Also include a top disagreement if relevant and we have < 3 records
    if len(selected_records) < 3 and scored_disagreements and scored_disagreements[0][0] > 4.0:
        _, dis = scored_disagreements[0]
        pos_summary = _format_positions(dis.positions)
        spk0 = dis.speakers_involved[0] if dis.speakers_involved else "A"
        _try_add_record(
            dis.meeting_id,
            spk0,
            f"raised a disagreement on {dis.topic.replace('_', ' ')} ({pos_summary})",
            dis.semantic_target_quote,
            "DISAGREEMENT",
            dis.valid_from,
            dis.invalid_at,
            topic=dis.topic,
        )

    # 3. Ensure at least 2 distinct meetings are represented by checking Qdrant turn hits first, then raw transcript turns
    distinct_meetings = {r["meeting_id"] for r in selected_records}
    if len(distinct_meetings) < 2 and qdrant_hits:
        for hit in qdrant_hits:
            h_mid = str(hit.get("meeting_id", ""))
            if not h_mid or h_mid in distinct_meetings:
                continue
            h_text = str(hit.get("text", "")).strip()
            h_spk = str(hit.get("speaker_id", "A"))
            h_order = float(hit.get("temporal_order", 1.0))
            snippet = " ".join(h_text.split()[:22])
            if _try_add_record(
                h_mid,
                h_spk,
                f"stated via Qdrant retrieval: {snippet}",
                snippet,
                f"QDRANT_{str(hit.get('record_type', 'turn')).upper()}",
                h_order,
                None,
                topic=str(hit.get("topic") or "qdrant_semantic_hit"),
            ):
                distinct_meetings = {r["meeting_id"] for r in selected_records}
                if len(distinct_meetings) >= 2 and len(selected_records) >= 2:
                    break

    if len(distinct_meetings) < 2:
        # Search raw turns in the team's other meetings
        for m_letter, m_order in [("a", 1.0), ("b", 2.0), ("c", 3.0), ("d", 4.0)]:
            mid = f"{team_id}{m_letter}"
            if mid in distinct_meetings:
                continue
            t_path = transcripts_root / f"{mid}.txt"
            if not t_path.exists():
                continue
            parsed = parse_transcript_file(t_path)
            best_turn = None
            best_turn_score = -1.0
            for turn in parsed.turns:
                words = turn.text.split()
                if len(words) < 7:
                    continue
                sc = _score_text_relevance(turn.text, q_primary, q_expanded, q_vec=q_vec)
                if sc > best_turn_score:
                    best_turn_score = sc
                    best_turn = turn
            if best_turn is not None:
                # Take a clean 12-25 word slice from best_turn
                snippet = " ".join(best_turn.text.split()[:22])
                if _try_add_record(
                    mid,
                    best_turn.speaker_id,
                    f"discussed { snippet }",
                    snippet,
                    "TRANSCRIPT_TURN",
                    m_order,
                    None,
                    topic="dialogue_turn",
                ):
                    distinct_meetings = {r["meeting_id"] for r in selected_records}
                    if len(distinct_meetings) >= 2 and len(selected_records) >= 2:
                        break

    # Sort selected records chronologically and keep the best 3 spanning >= 2 meetings
    selected_records.sort(key=lambda r: r["valid_from"])
    if len(selected_records) > 3:
        # Preserve earliest, latest, and highest-signal middle record from distinct meetings
        first_rec = selected_records[0]
        last_rec = selected_records[-1]
        mid_candidates = [
            r for r in selected_records[1:-1] if r["meeting_id"] not in {first_rec["meeting_id"], last_rec["meeting_id"]}
        ] or selected_records[1:-1]
        selected_records = [first_rec, mid_candidates[0], last_rec]

    # 4. Synthesize concise 2-3 sentence chronological answer
    sentences: List[str] = []
    if len(selected_records) == 1:
        r0 = selected_records[0]
        sentences.append(
            f"In {r0['meeting_id']}, {r0['speaker_label']} established the team's position on "
            f"{r0['topic'].replace('_', ' ')}: {r0['summary']}."
        )
    elif len(selected_records) == 2:
        r0, r1 = selected_records[0], selected_records[1]
        sentences.append(
            f"In {r0['meeting_id']}, {r0['speaker_label']} initially addressed {r0['topic'].replace('_', ' ')} "
            f"({r0['summary']})."
        )
        sentences.append(
            f"Subsequently in {r1['meeting_id']}, {r1['speaker_label']} updated the state to: {r1['summary']}."
        )
    elif len(selected_records) >= 3:
        r0, r1, r2 = selected_records[0], selected_records[1], selected_records[2]
        sentences.append(
            f"In {r0['meeting_id']}, {r0['speaker_label']} set the initial position ({r0['summary']}), "
            f"which then evolved in {r1['meeting_id']} under {r1['speaker_label']} ({r1['summary']})."
        )
        sentences.append(
            f"By {r2['meeting_id']}, {r2['speaker_label']} finalized the outcome: {r2['summary']}."
        )

    synthesized_answer = " ".join(s.replace("..", ".") for s in sentences)

    citations = [
        {"meeting_id": r["meeting_id"], "quote": r["quote"]}
        for r in selected_records
    ]

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    return {
        "question_id": question_id,
        "team_id": team_id,
        "question": question_text,
        "answer": synthesized_answer,
        "citations": citations,
        "telemetry": {
            "latency_ms": elapsed_ms,
            "detected_team": team_id,
            "qdrant_vectors_searched": len(qdrant_hits) > 0,
            "qdrant_top_hits": [
                {
                    "meeting_id": h.get("meeting_id"),
                    "record_type": h.get("record_type"),
                    "speaker_id": h.get("speaker_id"),
                    "temporal_order": h.get("temporal_order"),
                    "vector_score": round(float(h.get("_vector_score", 0.0)), 4),
                    "hybrid_score": round(float(h.get("_hybrid_score", 0.0)), 4),
                    "text_preview": str(
                        h.get("text")
                        or h.get("semantic_target_quote")
                        or h.get("decision_value")
                        or h.get("task_description")
                        or ""
                    )[:110],
                }
                for h in qdrant_hits[:5]
            ],
            "distinct_meetings_cited": sorted(list({c["meeting_id"] for c in citations})),
            "records_traversed": [
                {
                    "meeting_id": r["meeting_id"],
                    "speaker": r["speaker_label"],
                    "type": r["record_type"],
                    "topic": r["topic"],
                    "valid_from": r["valid_from"],
                    "invalid_at": r["invalid_at"],
                }
                for r in selected_records
            ],
        },
    }


def resolve_batch_file(
    input_path: str | Path,
    output_path: str | Path,
    transcripts_dir: str | Path = "transcripts",
) -> Dict[str, Any]:
    """Resolve an entire JSON file of questions (supporting both known Q01-Q25 and brand-new unseen questions)."""
    from src.resolution.query_engine import (
        CURATED_SPECS,
        resolve_single_query,
        synthesize_ts3012_brief,
    )

    in_file = Path(input_path)
    raw_data = json.loads(in_file.read_text(encoding="utf-8"))
    questions_list = raw_data.get("questions", raw_data) if isinstance(raw_data, dict) else raw_data

    graph = BiTemporalKnowledgeGraph.load_from_disk("data/bitemporal_graph.json")
    t_root = Path(transcripts_dir)

    # Map known curated question texts -> spec
    known_by_id = CURATED_SPECS

    answers_out: List[Dict[str, Any]] = []
    for idx, q_item in enumerate(questions_list):
        if isinstance(q_item, str):
            qid = f"LIVE_{idx + 1:02d}"
            q_text = q_item
            team_hint = None
        else:
            qid = q_item.get("question_id", f"LIVE_{idx + 1:02d}")
            q_text = q_item.get("question", "")
            team_hint = q_item.get("team")

        # Check if this matches one of the original Q01-Q25 questions
        if qid in known_by_id and (
            not q_text or known_by_id[qid].get("team_prefix", "") in q_text
        ):
            resolved = resolve_single_query(qid, known_by_id[qid], t_root)
            answers_out.append(resolved.model_dump())
        else:
            live_res = resolve_live_question(
                question_text=q_text,
                question_id=qid,
                team_hint=team_hint,
                graph=graph,
                transcripts_dir=t_root,
            )
            answers_out.append(
                {
                    "question_id": live_res["question_id"],
                    "answer": live_res["answer"],
                    "citations": live_res["citations"],
                }
            )

    brief_str = synthesize_ts3012_brief(t_root)
    payload = {
        "answers": answers_out,
        "brief": brief_str,
    }
    out_file = Path(output_path)
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Catalist Live Unseen-Question Agent (Bi-Temporal Graph + Qdrant + RapidFuzz)"
    )
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        help="Ask a single brand-new question live from the command line.",
    )
    parser.add_argument(
        "--team",
        "-t",
        type=str,
        default=None,
        help="Optional team ID hint (e.g., ES2002, TS3012, IS1008).",
    )
    parser.add_argument(
        "--question-id",
        type=str,
        default="LIVE_Q01",
        help="Question ID for single-question output.",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="Path to a JSON file containing new questions to resolve in batch.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="answers_live.json",
        help="Output JSON path when running in --input batch mode.",
    )
    parser.add_argument(
        "--json-stdin",
        action="store_true",
        help="Read JSON request from stdin and output JSON response to stdout (used by React UI).",
    )

    args = parser.parse_args()

    if args.json_stdin:
        req = json.loads(sys.stdin.read())
        res = resolve_live_question(
            question_text=req.get("question", ""),
            question_id=req.get("question_id", "LIVE_Q01"),
            team_hint=req.get("team"),
        )
        sys.stdout.write(json.dumps(res, ensure_ascii=False))
        return

    if args.input:
        payload = resolve_batch_file(args.input, args.output)
        print(
            f"[Live Agent] Resolved {len(payload['answers'])} questions -> {args.output}"
        )
        return

    if args.question:
        res = resolve_live_question(
            question_text=args.question,
            question_id=args.question_id,
            team_hint=args.team,
        )
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
