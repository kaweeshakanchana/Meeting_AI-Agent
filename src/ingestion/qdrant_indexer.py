"""Shared Qdrant vector store and payload indexer with filterable HNSW support.

Indexes parsed transcript turns/windows and subagent-extracted atomic decisions into
a persistent Qdrant collection (`ami_meeting_chunks`), enforcing payload schemas on:
- meeting_id (KEYWORD)
- team_id (KEYWORD)
- speaker_id (KEYWORD)
- temporal_order (FLOAT: a=1.0, b=2.0, c=3.0, d=4.0)
- record_type (KEYWORD: turn, window, decision, action, disagreement)
"""

from __future__ import annotations

import hashlib
import math
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.models.schemas import TeamExtractionBundle, TranscriptDocument

COLLECTION_NAME = "ami_meeting_chunks"
VECTOR_DIM = 384

TOKEN_RE = re.compile(r"[a-z0-9]+")

# Domain-important terms in the AMI corpus receive higher semantic weight during vector encoding
DOMAIN_CONCEPTS = {
    "lcd", "screen", "display", "touch", "touchscreen", "speech", "voice", "recognition",
    "microphone", "speaker", "chip", "print", "titanium", "plastic", "rubber", "latex",
    "wood", "wooden", "spongy", "fibreglass", "material", "case", "cover", "back",
    "battery", "batteries", "kinetic", "solar", "dynamo", "hand", "lithium", "rechargeable",
    "energy", "power", "cost", "costs", "price", "budget", "euro", "euros", "twelve",
    "half", "twenty", "five", "lost", "locator", "beep", "beeper", "whistle", "alarm",
    "signal", "light", "button", "buttons", "scroll", "wheel", "joystick", "trackball",
    "navigation", "menu", "teletext", "dvd", "vcr", "television", "tv", "only", "age",
    "young", "elderly", "children", " teenagers", "market", "target", "group", "rsi",
    "repetitive", "strain", "injury", "health", "ergonomic", "fruit", "vegetable",
    "banana", "pear", "apple", "tomato", "lemon", "shape", "curved", "double", "flat",
    "logo", "colour", "color", "yellow", "corporate", "slogan",
}

STOPWORDS = {
    "the", "and", "that", "this", "with", "for", "are", "was", "were", "have", "has",
    "had", "not", "but", "you", "your", "our", "they", "their", "them", "from", "what",
    "which", "who", "whom", "whose", "where", "when", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "than", "too", "very", "can",
    "will", "just", "should", "now", "okay", "yeah", "uh", "um", "mm", "hmm", "well",
    "like", "think", "know", "mean", "going", "gonna", "want", "would", "could", "about",
    "there", "here", "then", "also", "into", "out", "one", "two", "three",
}


def compute_dense_embedding(text: str, dim: int = VECTOR_DIM) -> List[float]:
    """Compute a deterministic L2-normalized 384-dim semantic + n-gram feature embedding."""
    vec = [0.0] * dim
    tokens = TOKEN_RE.findall(text.lower())
    if not tokens:
        vec[0] = 1.0
        return vec

    filtered = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    if not filtered:
        filtered = tokens

    # Unigrams + Bigrams + Character 4-grams for morphological resilience
    features: List[tuple[str, float]] = []
    for i, tok in enumerate(filtered):
        weight = 3.0 if tok in DOMAIN_CONCEPTS else 1.2
        features.append((f"u:{tok}", weight))
        if i + 1 < len(filtered):
            features.append((f"b:{tok}_{filtered[i + 1]}", weight * 0.9))
        if len(tok) >= 4:
            features.append((f"c:{tok[:4]}", 0.5))

    for feat, weight in features:
        digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
        h_int = int.from_bytes(digest, "little")
        idx = h_int % dim
        sign = 1.0 if ((h_int >> 32) & 1) == 0 else -1.0
        vec[idx] += sign * weight

    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 1e-9:
        vec = [v / norm for v in vec]
    else:
        vec[0] = 1.0
    return vec


class AMIHybridQdrantStore:
    """Manages the persistent Qdrant collection and hybrid search for AMI meeting transcripts."""

    def __init__(self, db_path: Path | str = "data/qdrant_db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(self.db_path))

    def initialize_collection(self, recreate: bool = False) -> None:
        """Create the Qdrant collection and configure payload indexes for filterable HNSW."""
        existing = [c.name for c in self.client.get_collections().collections]
        if COLLECTION_NAME in existing:
            if recreate:
                self.client.delete_collection(COLLECTION_NAME)
            else:
                return

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qmodels.VectorParams(
                size=VECTOR_DIM,
                distance=qmodels.Distance.COSINE,
            ),
        )

        # Configure payload schema indexes for filterable HNSW (ACORN traversal)
        for field_name, schema_type in [
            ("meeting_id", qmodels.PayloadSchemaType.KEYWORD),
            ("team_id", qmodels.PayloadSchemaType.KEYWORD),
            ("speaker_id", qmodels.PayloadSchemaType.KEYWORD),
            ("temporal_order", qmodels.PayloadSchemaType.FLOAT),
            ("record_type", qmodels.PayloadSchemaType.KEYWORD),
            ("topic", qmodels.PayloadSchemaType.KEYWORD),
        ]:
            try:
                self.client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name=field_name,
                    field_schema=schema_type,
                )
            except Exception:
                # Local Qdrant mode automatically indexes payload fields when create_payload_index is a no-op
                pass

    def index_transcript_documents(
        self,
        docs: Dict[str, TranscriptDocument],
        window_size: int = 6,
        window_stride: int = 3,
        batch_size: int = 256,
    ) -> int:
        """Index substantive speaker turns and multi-turn conversational windows into Qdrant."""
        points: List[qmodels.PointStruct] = []
        total_indexed = 0

        for meeting_id, doc in sorted(docs.items()):
            # 1. Index individual substantive speaker turns (>= 6 words so every turn is a valid citation candidate)
            for turn in doc.turns:
                if turn.word_count < 6:
                    continue
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"turn:{turn.turn_id}"))
                role = doc.speaker_roles.get(turn.speaker_id, "Unknown")
                embed_text = f"{doc.team_id} {doc.meeting_id} Speaker {turn.speaker_id} ({role}): {turn.text}"
                points.append(
                    qmodels.PointStruct(
                        id=point_id,
                        vector=compute_dense_embedding(embed_text),
                        payload={
                            "record_type": "turn",
                            "turn_id": turn.turn_id,
                            "meeting_id": doc.meeting_id,
                            "team_id": doc.team_id,
                            "meeting_letter": doc.meeting_letter,
                            "temporal_order": float(doc.temporal_order),
                            "timestamp": turn.timestamp,
                            "timestamp_seconds": float(turn.timestamp_seconds),
                            "speaker_id": turn.speaker_id,
                            "speaker_role": role,
                            "text": turn.text,
                            "word_count": turn.word_count,
                        },
                    )
                )

                if len(points) >= batch_size:
                    self.client.upsert(collection_name=COLLECTION_NAME, points=points)
                    total_indexed += len(points)
                    points.clear()

            # 2. Index multi-turn contextual windows for multi-speaker dialogue context
            substantive_turns = [t for t in doc.turns if t.word_count >= 3]
            for start_idx in range(0, len(substantive_turns), window_stride):
                win_turns = substantive_turns[start_idx : start_idx + window_size]
                if not win_turns:
                    continue
                window_id = f"{doc.meeting_id}_win_{start_idx:04d}"
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"window:{window_id}"))
                primary_speaker = win_turns[0].speaker_id
                window_lines = [
                    f"[{t.timestamp}] Speaker {t.speaker_id} ({doc.speaker_roles.get(t.speaker_id, 'Member')}): {t.text}"
                    for t in win_turns
                ]
                combined_text = "\n".join(window_lines)
                points.append(
                    qmodels.PointStruct(
                        id=point_id,
                        vector=compute_dense_embedding(combined_text),
                        payload={
                            "record_type": "window",
                            "window_id": window_id,
                            "meeting_id": doc.meeting_id,
                            "team_id": doc.team_id,
                            "meeting_letter": doc.meeting_letter,
                            "temporal_order": float(doc.temporal_order),
                            "timestamp": win_turns[0].timestamp,
                            "timestamp_seconds": float(win_turns[0].timestamp_seconds),
                            "speaker_id": primary_speaker,
                            "speakers_in_window": sorted({t.speaker_id for t in win_turns}),
                            "text": combined_text,
                            "turn_ids": [t.turn_id for t in win_turns],
                        },
                    )
                )

                if len(points) >= batch_size:
                    self.client.upsert(collection_name=COLLECTION_NAME, points=points)
                    total_indexed += len(points)
                    points.clear()

        if points:
            self.client.upsert(collection_name=COLLECTION_NAME, points=points)
            total_indexed += len(points)
            points.clear()

        return total_indexed

    def index_extraction_bundles(self, bundles: Sequence[TeamExtractionBundle]) -> int:
        """Index subagent-extracted atomic decisions, action items, and disagreements into Qdrant."""
        points: List[qmodels.PointStruct] = []
        for bundle in bundles:
            for dec in bundle.decisions:
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"decision:{dec.edge_id}"))
                embed_text = (
                    f"{dec.team_id} {dec.meeting_id} {dec.topic} {dec.status}: {dec.decision_value}. "
                    f"Rationale: {dec.rationale}. Quote: {dec.semantic_target_quote}"
                )
                points.append(
                    qmodels.PointStruct(
                        id=point_id,
                        vector=compute_dense_embedding(embed_text),
                        payload={
                            "record_type": "decision",
                            "edge_id": dec.edge_id,
                            "meeting_id": dec.meeting_id,
                            "team_id": dec.team_id,
                            "speaker_id": dec.speaker_id,
                            "speaker_role": dec.speaker_role or bundle.speaker_roles.get(dec.speaker_id, ""),
                            "temporal_order": float(dec.valid_from),
                            "valid_from": float(dec.valid_from),
                            "invalid_at": dec.invalid_at,
                            "topic": dec.topic,
                            "status": dec.status,
                            "decision_value": dec.decision_value,
                            "rationale": dec.rationale,
                            "text": dec.semantic_target_quote,
                        },
                    )
                )
            for act in bundle.action_items:
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"action:{act.action_id}"))
                embed_text = (
                    f"{act.team_id} {act.meeting_id} Action Owner Speaker {act.owner_speaker_id}: "
                    f"{act.task_description}. Resolution: {act.resolution_summary or ''} {act.semantic_target_quote}"
                )
                points.append(
                    qmodels.PointStruct(
                        id=point_id,
                        vector=compute_dense_embedding(embed_text),
                        payload={
                            "record_type": "action",
                            "action_id": act.action_id,
                            "meeting_id": act.meeting_id,
                            "team_id": act.team_id,
                            "speaker_id": act.owner_speaker_id,
                            "speaker_role": act.owner_role or bundle.speaker_roles.get(act.owner_speaker_id, ""),
                            "temporal_order": float(act.valid_from),
                            "status": act.status,
                            "task_description": act.task_description,
                            "resolution_summary": act.resolution_summary,
                            "text": act.semantic_target_quote,
                        },
                    )
                )
            for dis in bundle.disagreements:
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"disagreement:{dis.disagreement_id}"))
                embed_text = (
                    f"{dis.team_id} {dis.meeting_id} Disagreement on {dis.topic}: {dis.positions}. "
                    f"Quote: {dis.semantic_target_quote}"
                )
                points.append(
                    qmodels.PointStruct(
                        id=point_id,
                        vector=compute_dense_embedding(embed_text),
                        payload={
                            "record_type": "disagreement",
                            "disagreement_id": dis.disagreement_id,
                            "meeting_id": dis.meeting_id,
                            "team_id": dis.team_id,
                            "speaker_id": dis.speakers_involved[0] if dis.speakers_involved else "A",
                            "temporal_order": float(dis.valid_from),
                            "topic": dis.topic,
                            "positions": dis.positions,
                            "is_open": dis.is_open,
                            "text": dis.semantic_target_quote,
                        },
                    )
                )

        if points:
            self.client.upsert(collection_name=COLLECTION_NAME, points=points)
        return len(points)

    def hybrid_search(
        self,
        query: str,
        team_id: Optional[str] = None,
        meeting_id: Optional[str] = None,
        speaker_id: Optional[str] = None,
        record_type: Optional[str] = None,
        min_temporal_order: Optional[float] = None,
        max_temporal_order: Optional[float] = None,
        top_k: int = 15,
    ) -> List[Dict[str, Any]]:
        """Execute filterable HNSW vector search fused with lexical BM25-style keyword overlap (RRF)."""
        must_conditions: List[qmodels.Condition] = []
        if team_id:
            must_conditions.append(
                qmodels.FieldCondition(key="team_id", match=qmodels.MatchValue(value=team_id))
            )
        if meeting_id:
            must_conditions.append(
                qmodels.FieldCondition(key="meeting_id", match=qmodels.MatchValue(value=meeting_id))
            )
        if speaker_id:
            must_conditions.append(
                qmodels.FieldCondition(key="speaker_id", match=qmodels.MatchValue(value=speaker_id))
            )
        if record_type:
            must_conditions.append(
                qmodels.FieldCondition(key="record_type", match=qmodels.MatchValue(value=record_type))
            )
        if min_temporal_order is not None or max_temporal_order is not None:
            must_conditions.append(
                qmodels.FieldCondition(
                    key="temporal_order",
                    range=qmodels.Range(gte=min_temporal_order, lte=max_temporal_order),
                )
            )

        query_filter = qmodels.Filter(must=must_conditions) if must_conditions else None
        query_vec = compute_dense_embedding(query)

        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            query_filter=query_filter,
            limit=max(top_k * 3, 30),
            with_payload=True,
        )

        query_tokens = {t for t in TOKEN_RE.findall(query.lower()) if t not in STOPWORDS and len(t) > 1}
        results: List[Dict[str, Any]] = []
        for rank, scored_point in enumerate(response.points):
            payload = dict(scored_point.payload or {})
            text_tokens = set(TOKEN_RE.findall(str(payload.get("text", "")).lower()))
            lexical_overlap = len(query_tokens & text_tokens) / max(len(query_tokens), 1)
            rrf_score = (1.0 / (60.0 + rank + 1)) + 0.5 * float(scored_point.score) + 0.35 * lexical_overlap
            payload["_vector_score"] = float(scored_point.score)
            payload["_lexical_overlap"] = lexical_overlap
            payload["_hybrid_score"] = rrf_score
            results.append(payload)

        results.sort(key=lambda x: x["_hybrid_score"], reverse=True)
        return results[:top_k]

    def close(self) -> None:
        """Close the underlying Qdrant client cleanly."""
        self.client.close()
