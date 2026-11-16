"""Distributed Ingestion Orchestrator for Phase 1.

Coordinates:
1. Deterministic parsing of all 138 AMI transcripts across 35 teams.
2. Aggregation of parallel Antigravity subagent extraction bundles (`data/extracted/<TEAM_ID>.json`),
   supplemented by automated turn-level extraction for any remaining baseline teams.
3. Indexing of all speaker turns, sliding context windows, and extracted atomic facts into
   the shared Qdrant collection (`data/qdrant_db`) with filterable HNSW payload indexes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

from src.ingestion.parser import parse_all_transcripts
from src.ingestion.qdrant_indexer import AMIHybridQdrantStore
from src.models.schemas import (
    ActionItemNode,
    DecisionEdge,
    DisagreementNode,
    TeamExtractionBundle,
    TranscriptDocument,
)

TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "energy_source": ["battery", "batteries", "kinetic", "solar", "dynamo", "hand dynamo", "lithium", "power source", "energy"],
    "case_material": ["plastic", "rubber", "latex", "titanium", "wood", "wooden", "spongy", "material", "fibreglass", "casing"],
    "display_screen": ["lcd", "touch screen", "touchscreen", "display", "screen"],
    "speech_recognition": ["speech recognition", "voice recognition", "microphone", "voice sensor"],
    "locator_feature": ["lost", "locator", "beep", "beeper", "whistle", "alarm", "find the remote"],
    "navigation_control": ["scroll wheel", "joystick", "push button", "spinning wheel", "trackball", "navigation"],
    "production_cost_budget": ["cost", "budget", "twelve and a half", "12.5", "12.50", "euro", "euros", "expensive"],
    "selling_price": ["selling price", "twenty five", "25 euro", "retail", "profit"],
    "target_age_group": ["age group", "young", "elderly", "teenagers", "target market", "years old"],
    "shape_theme": ["fruit", "vegetable", "banana", "pear", "curved", "double curved", "ergonomic", "rsi", "repetitive strain"],
}


def build_baseline_bundle_for_team(
    team_id: str,
    team_docs: List[TranscriptDocument],
) -> TeamExtractionBundle:
    """Deterministically extract baseline decisions, actions, and disagreements for a team if not already created by a subagent."""
    team_docs_sorted = sorted(team_docs, key=lambda d: d.temporal_order)
    meetings_processed = [d.meeting_id for d in team_docs_sorted]
    speaker_roles: Dict[str, str] = {}
    for d in team_docs_sorted:
        speaker_roles.update(d.speaker_roles)

    decisions: List[DecisionEdge] = []
    action_items: List[ActionItemNode] = []
    disagreements: List[DisagreementNode] = []

    prior_topic_edge: Dict[str, DecisionEdge] = {}

    decision_cue_re = re.compile(
        r"\b(decide|decided|agree|agreed|going to use|we'll have|we will use|drop|dropped|keep|kept|too expensive|over budget|settled on|propose|suggest)\b",
        re.IGNORECASE,
    )
    action_cue_re = re.compile(
        r"\b(i'll look into|i will check|find out|check the price|look into that|next meeting)\b",
        re.IGNORECASE,
    )
    disagree_cue_re = re.compile(
        r"\b(disagree|don't think|too expensive|objection|not sure|prefer|instead of)\b",
        re.IGNORECASE,
    )

    for doc in team_docs_sorted:
        for turn in doc.turns:
            if turn.word_count < 6:
                continue
            lower_text = turn.text.lower()
            words = turn.text.split()
            quote_span = " ".join(words[: min(len(words), 22)])

            for topic, kw_list in TOPIC_KEYWORDS.items():
                if any(kw in lower_text for kw in kw_list):
                    if decision_cue_re.search(turn.text):
                        edge_id = f"{doc.meeting_id}_{topic}_{turn.turn_index}"
                        status = "DROPPED" if ("drop" in lower_text or "no " in lower_text or "too expensive" in lower_text) else "DECIDED"
                        supersedes_id = None
                        if topic in prior_topic_edge and prior_topic_edge[topic].meeting_id != doc.meeting_id:
                            prev_edge = prior_topic_edge[topic]
                            prev_edge.invalid_at = doc.temporal_order
                            supersedes_id = prev_edge.edge_id
                            if status == "DECIDED":
                                status = "CHANGED"

                        new_edge = DecisionEdge(
                            edge_id=edge_id,
                            team_id=team_id,
                            meeting_id=doc.meeting_id,
                            topic=topic,
                            decision_value=quote_span,
                            status=status,
                            speaker_id=turn.speaker_id,
                            speaker_role=speaker_roles.get(turn.speaker_id),
                            rationale=f"Extracted from turn {turn.turn_id} at {turn.timestamp}",
                            valid_from=doc.temporal_order,
                            invalid_at=None,
                            supersedes_edge_id=supersedes_id,
                            semantic_target_quote=quote_span,
                        )
                        decisions.append(new_edge)
                        prior_topic_edge[topic] = new_edge
                        break

            if action_cue_re.search(turn.text) and len(action_items) < 12:
                action_items.append(
                    ActionItemNode(
                        action_id=f"{doc.meeting_id}_act_{turn.turn_index}",
                        team_id=team_id,
                        meeting_id=doc.meeting_id,
                        owner_speaker_id=turn.speaker_id,
                        owner_role=speaker_roles.get(turn.speaker_id),
                        task_description=quote_span,
                        status="OPEN",
                        valid_from=doc.temporal_order,
                        semantic_target_quote=quote_span,
                    )
                )

            if disagree_cue_re.search(turn.text) and len(disagreements) < 10:
                disagreements.append(
                    DisagreementNode(
                        disagreement_id=f"{doc.meeting_id}_dis_{turn.turn_index}",
                        team_id=team_id,
                        meeting_id=doc.meeting_id,
                        topic="design_tradeoff",
                        positions=quote_span,
                        speakers_involved=[turn.speaker_id],
                        is_open=(doc.meeting_letter == "d"),
                        valid_from=doc.temporal_order,
                        semantic_target_quote=quote_span,
                    )
                )

    return TeamExtractionBundle(
        team_id=team_id,
        meetings_processed=meetings_processed,
        speaker_roles=speaker_roles,
        decisions=decisions,
        action_items=action_items,
        disagreements=disagreements,
    )


def run_phase1_ingestion(
    transcripts_dir: Path | str = "transcripts",
    extracted_dir: Path | str = "data/extracted",
    qdrant_dir: Path | str = "data/qdrant_db",
    recreate_qdrant: bool = True,
) -> Dict[str, int]:
    """Execute the full Phase 1 ingestion pipeline across all 138 transcripts and extracted bundles."""
    extracted_path = Path(extracted_dir)
    extracted_path.mkdir(parents=True, exist_ok=True)

    docs = parse_all_transcripts(transcripts_dir)
    teams_map: Dict[str, List[TranscriptDocument]] = {}
    for doc in docs.values():
        teams_map.setdefault(doc.team_id, []).append(doc)

    bundles: List[TeamExtractionBundle] = []
    for team_id, team_docs in sorted(teams_map.items()):
        bundle_file = extracted_path / f"{team_id}.json"
        if bundle_file.exists():
            try:
                raw_data = json.loads(bundle_file.read_text(encoding="utf-8"))
                bundle = TeamExtractionBundle.model_validate(raw_data)
                bundles.append(bundle)
                continue
            except Exception as exc:
                print(f"[WARN] Rebuilding invalid bundle for {team_id}: {exc}")

        bundle = build_baseline_bundle_for_team(team_id, team_docs)
        bundle_file.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
        bundles.append(bundle)

    store = AMIHybridQdrantStore(db_path=qdrant_dir)
    store.initialize_collection(recreate=recreate_qdrant)
    chunks_indexed = store.index_transcript_documents(docs)
    facts_indexed = store.index_extraction_bundles(bundles)
    store.close()

    stats = {
        "transcripts_parsed": len(docs),
        "teams_processed": len(teams_map),
        "turns_and_windows_indexed": chunks_indexed,
        "atomic_facts_indexed": facts_indexed,
    }
    return stats


if __name__ == "__main__":
    summary = run_phase1_ingestion()
    print(json.dumps(summary, indent=2))
