"""Pydantic schemas for transcript parsing, Qdrant indexing, Bi-Temporal Knowledge Graph, and answers.json serialization."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


MEETING_ORDER_MAP: Dict[str, float] = {
    "a": 1.0,
    "b": 2.0,
    "c": 3.0,
    "d": 4.0,
}


class SpeakerTurn(BaseModel):
    """Represents a single unedited speaker turn parsed deterministically from an AMI transcript."""

    turn_id: str = Field(..., description="Unique turn ID, e.g., ES2002a_0001")
    meeting_id: str = Field(..., description="Meeting identifier, e.g., ES2002a")
    team_id: str = Field(..., description="Team identifier, e.g., ES2002")
    meeting_letter: str = Field(..., description="Meeting sequence letter: a, b, c, or d")
    temporal_order: float = Field(..., description="Float representation of meeting sequence: a=1.0, b=2.0, c=3.0, d=4.0")
    turn_index: int = Field(..., description="0-based index of the turn within the meeting")
    timestamp: str = Field(..., description="Turn start timestamp HH:MM:SS")
    timestamp_seconds: float = Field(..., description="Turn start timestamp converted to seconds")
    speaker_id: str = Field(..., description="Speaker letter: A, B, C, or D")
    text: str = Field(..., description="Exact unedited text of the speaker turn preserving disfluencies")
    word_count: int = Field(..., description="Number of whitespace-delimited words in the turn")


class TranscriptDocument(BaseModel):
    """Represents a parsed AMI meeting transcript file."""

    meeting_id: str
    team_id: str
    meeting_letter: str
    temporal_order: float
    speakers: List[str]
    speaker_roles: Dict[str, str] = Field(default_factory=dict)
    speaker_names: Dict[str, str] = Field(default_factory=dict)
    turns: List[SpeakerTurn]
    raw_text: str


class DecisionEdge(BaseModel):
    """Bi-Temporal Knowledge Graph edge tracking a project decision, proposal, cost, or state transition."""

    edge_id: str = Field(..., description="Unique identifier for the bi-temporal edge")
    team_id: str = Field(..., description="Team ID, e.g., TS3012")
    meeting_id: str = Field(..., description="Meeting ID where this state was recorded, e.g., TS3012a")
    topic: str = Field(..., description="Canonical topic slug, e.g., case_material, energy_source, lcd_screen")
    decision_value: str = Field(..., description="Concise summary of the decision, proposal, or figure")
    status: str = Field(
        default="DECIDED",
        description="Status: PROPOSED, PROVISIONAL, DECIDED, CHANGED, DROPPED, REJECTED, KEPT",
    )
    speaker_id: str = Field(..., description="Speaker letter (A, B, C, D) who stated or proposed this")
    speaker_role: Optional[str] = Field(default=None, description="Role of the speaker if known")
    rationale: str = Field(default="", description="Reason, objection, or budget constraint motivating this state")
    valid_from: float = Field(..., description="Chronological timestamp when fact became valid (e.g. 1.0, 2.0, 3.0, 4.0)")
    invalid_at: Optional[float] = Field(
        default=None,
        description="Chronological timestamp when fact was superseded (None if still current on temporal frontier)",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="System ingestion timestamp (ISO 8601)",
    )
    deleted_at: Optional[str] = Field(default=None, description="Rollback timestamp if logically deleted")
    supersedes_edge_id: Optional[str] = Field(
        default=None, description="ID of the prior DecisionEdge that this decision supersedes"
    )
    semantic_target_quote: str = Field(
        ...,
        description="Target quote (>= 6 words from a single speaker turn) to be verified via deterministic RapidFuzz matching",
    )


class ActionItemNode(BaseModel):
    """Tracks commitments, follow-ups, component price checks, and open actions across meetings."""

    action_id: str
    team_id: str
    meeting_id: str
    owner_speaker_id: str
    owner_role: Optional[str] = None
    task_description: str
    status: str = Field(default="OPEN", description="OPEN, REPORTED_BACK, COMPLETED, DROPPED")
    resolution_meeting_id: Optional[str] = None
    resolution_summary: Optional[str] = None
    valid_from: float
    invalid_at: Optional[float] = None
    semantic_target_quote: str


class DisagreementNode(BaseModel):
    """Tracks objections, pushbacks, and unresolved disagreements within a team."""

    disagreement_id: str
    team_id: str
    meeting_id: str
    topic: str
    positions: str
    speakers_involved: List[str]
    is_open: bool = True
    valid_from: float
    invalid_at: Optional[float] = None
    semantic_target_quote: str


class TeamExtractionBundle(BaseModel):
    """Complete structured extraction for a team across its chronological meetings."""

    team_id: str
    meetings_processed: List[str]
    speaker_roles: Dict[str, str] = Field(default_factory=dict)
    speaker_names: Dict[str, str] = Field(default_factory=dict)
    decisions: List[DecisionEdge] = Field(default_factory=list)
    action_items: List[ActionItemNode] = Field(default_factory=list)
    disagreements: List[DisagreementNode] = Field(default_factory=list)


class Citation(BaseModel):
    """Exact verbatim citation verified against the raw transcript."""

    meeting_id: str = Field(..., description="Meeting ID, e.g., ES2008b")
    quote: str = Field(..., description="Exact verbatim words (>= 6 words) from a single speaker turn")


class QuestionAnswer(BaseModel):
    """Single question answer entry in answers.json."""

    question_id: str = Field(..., description="Question ID, e.g., Q01")
    answer: str = Field(..., description="1 to 3 plain sentences answering the question")
    citations: List[Citation] = Field(..., min_length=1, max_length=3)


class SubmissionOutput(BaseModel):
    """Root schema for answers.json required by the Catalist build test."""

    answers: List[QuestionAnswer]
    brief: str
