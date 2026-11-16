"""Deterministic Python parser for AMI Meeting Corpus transcripts.

Strictly enforces the Model-vs-Code boundary: splitting speaker turns, parsing timestamps,
mapping meeting letters (a-d) to temporal_order floats (1.0-4.0), and retaining disfluencies
("um", "uh") are executed in 100% deterministic Python without touching an LLM.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.models.schemas import MEETING_ORDER_MAP, SpeakerTurn, TranscriptDocument

TURN_HEADER_RE = re.compile(r"^(\d{2}:\d{2}:\d{2})\s+Speaker\s+([A-D])\s*$")

ROLE_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    (
        "Project Manager",
        re.compile(
            r"\b(?:i'm|i am|i'll be|i will be|act as|playing the|role of|my role is)\b[^.?!]{0,50}\b(?:project manager|pm)\b"
            r"|\bproject manager\b[^.?!]{0,30}\b(?:that's me|speaking)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Industrial Designer",
        re.compile(
            r"\b(?:i'm|i am|i'll be|i will be|supposed to be|act as|role of)\b[^.?!]{0,50}\b(?:industrial designer|id)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "User Interface Designer",
        re.compile(
            r"\b(?:i'm|i am|i'll be|i will be|supposed to be|act as|role of)\b[^.?!]{0,50}\b(?:user interface(?: designer)?|ui designer|interface designer|uid)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Marketing Expert",
        re.compile(
            r"\b(?:i'm|i am|i'll be|i will be|supposed to be|act as|role of)\b[^.?!]{0,50}\b(?:marketing(?: expert| manager)?|me)\b",
            re.IGNORECASE,
        ),
    ),
]


def timestamp_to_seconds(ts: str) -> float:
    """Convert HH:MM:SS string into float seconds."""
    parts = ts.strip().split(":")
    if len(parts) != 3:
        return 0.0
    hours, minutes, seconds = int(parts[0]), int(parts[1]), float(parts[2])
    return float(hours * 3600 + minutes * 60 + seconds)


def extract_speaker_roles_from_turns(turns: List[SpeakerTurn]) -> Dict[str, str]:
    """Deterministically detect self-introduced speaker roles from early meeting turns."""
    roles: Dict[str, str] = {}
    # Check first 60 turns where introductions typically happen
    for turn in turns[:60]:
        if turn.speaker_id in roles:
            continue
        for role_name, pattern in ROLE_PATTERNS:
            if pattern.search(turn.text):
                if role_name not in roles.values():
                    roles[turn.speaker_id] = role_name
                    break
    return roles


def parse_transcript_file(file_path: Path | str) -> TranscriptDocument:
    """Parse a raw AMI transcript .txt file into a structured TranscriptDocument."""
    path = Path(file_path)
    raw_text = path.read_text(encoding="utf-8")
    lines = raw_text.splitlines()

    meeting_id = path.stem
    team_id = meeting_id[:-1]
    meeting_letter = meeting_id[-1].lower()
    temporal_order = MEETING_ORDER_MAP.get(meeting_letter, 1.0)
    speakers: List[str] = ["A", "B", "C", "D"]

    # Parse metadata header lines at the top of the file
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            break
        if line.startswith("Meeting:"):
            meeting_id = line.split(":", 1)[1].strip()
            team_id = meeting_id[:-1]
            meeting_letter = meeting_id[-1].lower()
            temporal_order = MEETING_ORDER_MAP.get(meeting_letter, 1.0)
        elif line.startswith("Team:"):
            team_id = line.split(":", 1)[1].strip()
        elif line.startswith("Speakers:"):
            raw_speakers = line.split(":", 1)[1].strip()
            parsed_sp = [
                s.replace("Speaker", "").strip()
                for s in raw_speakers.split(",")
                if s.strip()
            ]
            if parsed_sp:
                speakers = parsed_sp
        idx += 1

    turns: List[SpeakerTurn] = []
    current_ts: Optional[str] = None
    current_speaker: Optional[str] = None
    current_lines: List[str] = []
    turn_index = 0

    def flush_turn() -> None:
        nonlocal turn_index, current_ts, current_speaker, current_lines
        if current_ts is not None and current_speaker is not None:
            turn_text = "\n".join(current_lines).strip()
            if turn_text:
                words = turn_text.split()
                turns.append(
                    SpeakerTurn(
                        turn_id=f"{meeting_id}_{turn_index:04d}",
                        meeting_id=meeting_id,
                        team_id=team_id,
                        meeting_letter=meeting_letter,
                        temporal_order=temporal_order,
                        turn_index=turn_index,
                        timestamp=current_ts,
                        timestamp_seconds=timestamp_to_seconds(current_ts),
                        speaker_id=current_speaker,
                        text=turn_text,
                        word_count=len(words),
                    )
                )
                turn_index += 1
        current_ts = None
        current_speaker = None
        current_lines = []

    for line in lines[idx:]:
        match = TURN_HEADER_RE.match(line.strip())
        if match:
            flush_turn()
            current_ts = match.group(1)
            current_speaker = match.group(2)
        else:
            if current_ts is not None:
                if line.strip() or current_lines:
                    current_lines.append(line)

    flush_turn()
    speaker_roles = extract_speaker_roles_from_turns(turns)

    return TranscriptDocument(
        meeting_id=meeting_id,
        team_id=team_id,
        meeting_letter=meeting_letter,
        temporal_order=temporal_order,
        speakers=speakers,
        speaker_roles=speaker_roles,
        turns=turns,
        raw_text=raw_text,
    )


def parse_all_transcripts(transcripts_dir: Path | str) -> Dict[str, TranscriptDocument]:
    """Parse all .txt transcript files in the given directory sorted chronologically."""
    base_dir = Path(transcripts_dir)
    docs: Dict[str, TranscriptDocument] = {}
    for file_path in sorted(base_dir.glob("*.txt")):
        doc = parse_transcript_file(file_path)
        docs[doc.meeting_id] = doc

    # Propagate discovered speaker roles across all meetings of the same team
    team_roles: Dict[str, Dict[str, str]] = {}
    for doc in docs.values():
        if doc.team_id not in team_roles:
            team_roles[doc.team_id] = {}
        for spk, role in doc.speaker_roles.items():
            if spk not in team_roles[doc.team_id]:
                team_roles[doc.team_id][spk] = role

    for doc in docs.values():
        merged = dict(team_roles.get(doc.team_id, {}))
        merged.update(doc.speaker_roles)
        doc.speaker_roles = merged

    return docs
