"""Recovers and normalizes all subagent-written TeamExtractionBundle JSON files from subagent transcripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.models.schemas import (
    ActionItemNode,
    DecisionEdge,
    DisagreementNode,
    MEETING_ORDER_MAP,
    TeamExtractionBundle,
)

SUBAGENT_IDS = [
    "0399da22-9990-462c-976f-62ddd15162bc",
    "d813ea24-f731-4a45-81c9-f0f8c63fa5f4",
    "353e459c-3742-49b2-b4a7-b65cfcc1a2d4",
    "9d8f6f21-eee5-40a5-b686-f2361666bec4",
    "99a889e9-7ac7-49e6-8023-c8516e5272d3",
]

BRAIN_DIR = Path(r"C:\Users\YOGA\.gemini\antigravity\brain")
EXTRACTED_DIR = Path(r"d:\Catalist_Project\data\extracted")


def infer_valid_from(meeting_id: str, fallback: float = 1.0) -> float:
    if meeting_id and meeting_id[-1].lower() in MEETING_ORDER_MAP:
        return MEETING_ORDER_MAP[meeting_id[-1].lower()]
    return float(fallback)


def normalize_bundle_dict(team_id: str, raw: Dict[str, Any]) -> TeamExtractionBundle:
    raw_meetings = raw.get("meetings_processed") or raw.get("meetings")
    if isinstance(raw_meetings, list) and raw_meetings and isinstance(raw_meetings[0], dict):
        meetings = [str(m.get("meeting_id") or f"{team_id}a") for m in raw_meetings]
    elif isinstance(raw_meetings, list) and raw_meetings:
        meetings = [str(m) for m in raw_meetings]
    else:
        meetings = [f"{team_id}a", f"{team_id}b", f"{team_id}c", f"{team_id}d"]

    speaker_roles: Dict[str, str] = {}
    speaker_names: Dict[str, str] = {}

    if isinstance(raw.get("speaker_roles"), dict):
        for k, v in raw["speaker_roles"].items():
            speaker_roles[str(k).replace("Speaker ", "").strip()[-1]] = str(v)
    if isinstance(raw.get("speaker_names"), dict):
        for k, v in raw["speaker_names"].items():
            speaker_names[str(k).replace("Speaker ", "").strip()[-1]] = str(v)

    # Also support `speakers` list or dict
    raw_speakers = raw.get("speakers")
    if isinstance(raw_speakers, dict):
        for k, v in raw_speakers.items():
            spk_key = str(k).replace("Speaker ", "").strip()[-1]
            if isinstance(v, dict):
                if v.get("role"):
                    speaker_roles[spk_key] = str(v["role"])
                if v.get("name"):
                    speaker_names[spk_key] = str(v["name"])
            else:
                speaker_roles[spk_key] = str(v)
    elif isinstance(raw_speakers, list):
        for sp in raw_speakers:
            if isinstance(sp, dict):
                spk_key = str(sp.get("speaker_letter") or sp.get("speaker_id") or "A")[-1]
                if sp.get("role"):
                    speaker_roles[spk_key] = str(sp["role"])
                if sp.get("name") or sp.get("GiveName"):
                    speaker_names[spk_key] = str(sp.get("name") or sp.get("GiveName"))

    # Collect raw decision candidates from `decisions`, `edges`, or `meetings[*].edges`
    raw_decisions: List[Dict[str, Any]] = list(raw.get("decisions") or [])
    if not raw_decisions and isinstance(raw.get("edges"), list):
        raw_decisions = list(raw["edges"])
    if not raw_decisions and isinstance(raw.get("meetings"), list):
        for m_obj in raw["meetings"]:
            if isinstance(m_obj, dict):
                for sp in m_obj.get("speakers") or []:
                    if isinstance(sp, dict):
                        spk_key = str(sp.get("speaker_letter") or sp.get("speaker_id") or "A")[-1]
                        if sp.get("role"):
                            speaker_roles[spk_key] = str(sp["role"])
                        if sp.get("name"):
                            speaker_names[spk_key] = str(sp["name"])
                for e_obj in m_obj.get("edges") or []:
                    if isinstance(e_obj, dict):
                        raw_decisions.append(e_obj)

    decisions: List[DecisionEdge] = []
    for idx, d in enumerate(raw_decisions):
        cit = d.get("citation") if isinstance(d.get("citation"), dict) else {}
        m_id = str(d.get("meeting_id") or d.get("source_meeting") or cit.get("meeting_id") or f"{team_id}a")
        vf = d.get("valid_from")
        if vf is None:
            vf = infer_valid_from(m_id)
        spk_raw = str(
            d.get("speaker_id")
            or d.get("speaker_letter")
            or d.get("speaker")
            or d.get("proposer")
            or d.get("source_speaker_id")
            or cit.get("speaker")
            or "A"
        ).replace("Speaker ", "").strip()
        spk_id = spk_raw[-1] if spk_raw else "A"
        quote_str = str(
            d.get("semantic_target_quote")
            or d.get("quote")
            or d.get("target_quote")
            or cit.get("semantic_target_quote")
            or cit.get("quote")
            or ""
        )
        val_str = str(
            d.get("decision_value")
            or d.get("decision_summary")
            or d.get("decision")
            or d.get("value")
            or d.get("summary")
            or d.get("description")
            or quote_str
        )
        topic_str = str(
            d.get("topic")
            or d.get("category")
            or d.get("target_entity_id")
            or d.get("relation_type")
            or "general"
        )
        decisions.append(
            DecisionEdge(
                edge_id=str(d.get("edge_id") or d.get("decision_id") or d.get("id") or f"{team_id}_dec_{idx:03d}"),
                team_id=str(d.get("team_id") or team_id),
                meeting_id=m_id,
                topic=topic_str,
                decision_value=val_str,
                status=str(d.get("status") or d.get("decision_status") or "DECIDED").upper(),
                speaker_id=spk_id,
                speaker_role=d.get("speaker_role") or speaker_roles.get(spk_id),
                rationale=str(d.get("rationale") or d.get("reason") or ""),
                valid_from=float(vf),
                invalid_at=float(d["invalid_at"]) if d.get("invalid_at") is not None else None,
                supersedes_edge_id=d.get("supersedes_edge_id") or d.get("supersedes"),
                semantic_target_quote=quote_str,
            )
        )

    action_items: List[ActionItemNode] = []
    for idx, a in enumerate(raw.get("action_items", [])):
        cit = a.get("citation") if isinstance(a.get("citation"), dict) else {}
        m_id = str(a.get("meeting_id") or a.get("source_meeting") or cit.get("meeting_id") or f"{team_id}a")
        vf = a.get("valid_from")
        if vf is None:
            vf = infer_valid_from(m_id)
        owner_raw = str(
            a.get("owner_speaker_id") or a.get("owner") or a.get("speaker_id") or cit.get("speaker") or "A"
        ).replace("Speaker ", "").strip()
        owner = owner_raw[-1] if owner_raw else "A"
        quote_str = str(
            a.get("semantic_target_quote")
            or a.get("quote")
            or a.get("target_quote")
            or cit.get("semantic_target_quote")
            or cit.get("quote")
            or ""
        )
        inv_at = a.get("invalid_at") if a.get("invalid_at") is not None else a.get("resolved_at")
        action_items.append(
            ActionItemNode(
                action_id=str(a.get("action_id") or a.get("node_id") or a.get("id") or f"{team_id}_act_{idx:03d}"),
                team_id=str(a.get("team_id") or team_id),
                meeting_id=m_id,
                owner_speaker_id=owner,
                owner_role=a.get("owner_role") or speaker_roles.get(owner),
                task_description=str(a.get("task_description") or a.get("description") or a.get("task") or ""),
                status=str(a.get("status") or "OPEN"),
                resolution_meeting_id=a.get("resolution_meeting_id") or a.get("resolved_in"),
                resolution_summary=a.get("resolution_summary") or a.get("resolution") or a.get("outcome"),
                valid_from=float(vf),
                invalid_at=float(inv_at) if inv_at is not None else None,
                semantic_target_quote=quote_str,
            )
        )

    disagreements: List[DisagreementNode] = []
    for idx, dis in enumerate(raw.get("disagreements", [])):
        cit = dis.get("citation") if isinstance(dis.get("citation"), dict) else {}
        m_id = str(dis.get("meeting_id") or dis.get("source_meeting") or cit.get("meeting_id") or f"{team_id}a")
        vf = dis.get("valid_from")
        if vf is None:
            vf = infer_valid_from(m_id)
        spk_list = dis.get("speakers_involved") or dis.get("speakers")
        if not spk_list:
            init_s = dis.get("Initiator") or dis.get("initiator")
            cnt_s = dis.get("counter_party")
            spk_list = [s for s in [init_s, cnt_s] if s] or ["A"]
        if isinstance(spk_list, str):
            spk_list = [spk_list]
        spk_list = [str(s).replace("Speaker ", "").strip()[-1] for s in spk_list if str(s).strip()]
        quote_str = str(
            dis.get("semantic_target_quote")
            or dis.get("quote")
            or dis.get("target_quote")
            or cit.get("semantic_target_quote")
            or cit.get("quote")
            or ""
        )
        is_open_val = dis.get("is_open")
        if is_open_val is None:
            if "resolved" in dis:
                is_open_val = not bool(dis["resolved"])
            else:
                is_open_val = str(dis.get("status", "OPEN")).upper() == "OPEN"
        disagreements.append(
            DisagreementNode(
                disagreement_id=str(
                    dis.get("disagreement_id") or dis.get("node_id") or dis.get("id") or f"{team_id}_dis_{idx:03d}"
                ),
                team_id=str(dis.get("team_id") or team_id),
                meeting_id=m_id,
                topic=str(dis.get("topic") or "design_disagreement"),
                positions=str(
                    dis.get("positions")
                    or dis.get("objection_summary")
                    or dis.get("description")
                    or dis.get("summary")
                    or ""
                ),
                speakers_involved=spk_list,
                is_open=bool(is_open_val),
                valid_from=float(vf),
                invalid_at=float(dis["invalid_at"]) if dis.get("invalid_at") is not None else None,
                semantic_target_quote=quote_str,
            )
        )

    return TeamExtractionBundle(
        team_id=team_id,
        meetings_processed=[str(m) for m in meetings],
        speaker_roles=speaker_roles,
        speaker_names=speaker_names,
        decisions=decisions,
        action_items=action_items,
        disagreements=disagreements,
    )


def recover_and_normalize_all() -> List[str]:
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    recovered_teams: List[str] = []

    for sub_id in SUBAGENT_IDS:
        full_log = BRAIN_DIR / sub_id / ".system_generated" / "logs" / "transcript_full.jsonl"
        if not full_log.exists():
            full_log = BRAIN_DIR / sub_id / ".system_generated" / "logs" / "transcript.jsonl"
        if not full_log.exists():
            continue

        for line in full_log.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                step = json.loads(line)
            except Exception:
                continue
            for tc in step.get("tool_calls") or []:
                if tc.get("name") == "write_file" or tc.get("tool_name") == "write_file":
                    args = tc.get("arguments") or tc.get("args") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            continue
                    target_path = str(args.get("Path") or args.get("path") or "")
                    content = str(args.get("Content") or args.get("content") or "")
                    if "extracted" in target_path.lower() and target_path.endswith(".json") and content:
                        team_id = Path(target_path).stem
                        try:
                            raw_dict = json.loads(content)
                            bundle = normalize_bundle_dict(team_id, raw_dict)
                            out_file = EXTRACTED_DIR / f"{team_id}.json"
                            out_file.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
                            if team_id not in recovered_teams:
                                recovered_teams.append(team_id)
                        except Exception as exc:
                            print(f"[ERROR] Failed normalizing {team_id}: {exc}")

    return sorted(recovered_teams)


if __name__ == "__main__":
    teams = recover_and_normalize_all()
    print(f"Normalized {len(teams)} subagent team bundles: {teams}")
