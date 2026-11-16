"""Deterministic Verbatim Citation Extractor (Phase 3 — Model-vs-Code Boundary).

Implements `.agents/skills/verbatim-citation/SKILL.md`:
- The LLM is strictly prohibited from emitting the final quote string directly into `answers.json`.
- `locate_verbatim_quote(transcript_path, target_quote, threshold=95)` normalizes both the raw
  transcript speaker turns and the semantic target quote (lowercase, strip punctuation, collapse
  whitespace), computes optimal substring alignment via `rapidfuzz.fuzz.partial_ratio_alignment`,
  and slices the exact, un-normalized words from a single speaker turn in the source file.
- If the alignment score falls below `threshold` (default 95), raises `CitationHallucinationError`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from rapidfuzz import fuzz

from src.ingestion.parser import parse_transcript_file
from src.models.schemas import SpeakerTurn, TeamExtractionBundle, TranscriptDocument


class CitationHallucinationError(ValueError):
    """Raised when an LLM semantic quote fails deterministic RapidFuzz verification against the raw transcript."""


_TRANSCRIPT_CACHE: Dict[str, TranscriptDocument] = {}


def _get_cached_transcript(transcript_path: Path | str) -> TranscriptDocument:
    key = str(Path(transcript_path).resolve())
    if key not in _TRANSCRIPT_CACHE:
        _TRANSCRIPT_CACHE[key] = parse_transcript_file(transcript_path)
    return _TRANSCRIPT_CACHE[key]


DISFLUENCY_TOKENS = {"um", "uh", "er", "ah", "hmm", "mm", "mhm", "uhhuh", "mmhmm"}


def normalize_with_char_map(text: str, strip_disfluencies: bool = True) -> Tuple[str, List[int]]:
    """Normalize text (lowercase, strip punctuation and filler disfluencies, collapse whitespace)
    while retaining an exact character index map back to `text`.

    Because `char_map` stores the original character indices in `text`, slicing `text[raw_start:raw_end]`
    after alignment automatically restores all original disfluencies ("um", "uh"), punctuation, and
    AMI acronym underscores ("L_C_D_", "T_V_") that occur within the matched span.
    """
    # Tokenize into alphanumeric words with their exact character spans in `text`
    word_spans: List[Tuple[str, List[int]]] = []
    curr_chars: List[str] = []
    curr_indices: List[int] = []

    for idx, ch in enumerate(text):
        if ch.isalnum():
            curr_chars.append(ch.lower())
            curr_indices.append(idx)
        elif ch in ("_", "'", "-"):
            # Skip intra-word punctuation (so L_C_D_ -> lcd, don't -> dont) without breaking the token
            continue
        else:
            if curr_chars:
                word = "".join(curr_chars)
                if not (strip_disfluencies and word in DISFLUENCY_TOKENS):
                    word_spans.append((word, curr_indices))
                curr_chars = []
                curr_indices = []

    if curr_chars:
        word = "".join(curr_chars)
        if not (strip_disfluencies and word in DISFLUENCY_TOKENS):
            word_spans.append((word, curr_indices))

    norm_chars: List[str] = []
    char_map: List[int] = []
    for w_idx, (word, indices) in enumerate(word_spans):
        if w_idx > 0:
            norm_chars.append(" ")
            char_map.append(indices[0])
        for ch, orig_i in zip(word, indices):
            norm_chars.append(ch)
            char_map.append(orig_i)

    return "".join(norm_chars), char_map


def _snap_to_word_boundaries(raw_text: str, start_idx: int, end_idx: int, min_words: int = 6) -> str:
    """Expand character slice `[start_idx, end_idx]` to full whitespace-delimited tokens and enforce `min_words`."""
    n = len(raw_text)
    start_idx = max(0, min(start_idx, n - 1))
    end_idx = max(start_idx + 1, min(end_idx, n))

    # Snap start_idx left to start of current token
    while start_idx > 0 and not raw_text[start_idx - 1].isspace():
        start_idx -= 1

    # Snap end_idx right to end of current token
    while end_idx < n and not raw_text[end_idx].isspace():
        end_idx += 1

    # Collapse internal newlines to spaces if any, or slice raw_text directly
    # Wait: in AMI transcripts, each speaker turn is a single line, so raw_text[start_idx:end_idx] is contiguous
    candidate = raw_text[start_idx:end_idx].strip()
    words = candidate.split()

    if len(words) >= min_words:
        return candidate

    # Expand outward within the same speaker turn until we have at least `min_words`
    all_tokens = list(re.finditer(r"\S+", raw_text))
    if len(all_tokens) < min_words:
        return raw_text.strip()

    # Find token indices overlapping [start_idx, end_idx]
    first_tok = 0
    last_tok = len(all_tokens) - 1
    for i, m in enumerate(all_tokens):
        if m.end() > start_idx:
            first_tok = i
            break
    for i in range(len(all_tokens) - 1, -1, -1):
        if all_tokens[i].start() < end_idx:
            last_tok = i
            break

    while (last_tok - first_tok + 1) < min_words:
        if last_tok + 1 < len(all_tokens):
            last_tok += 1
        elif first_tok > 0:
            first_tok -= 1
        else:
            break

    new_start = all_tokens[first_tok].start()
    new_end = all_tokens[last_tok].end()
    return raw_text[new_start:new_end].strip()


def locate_verbatim_quote(
    transcript_path: Path | str,
    target_quote: str,
    threshold: float = 95.0,
    min_words: int = 6,
) -> str:
    """Locate and extract the exact, un-normalized verbatim quote from `transcript_path` using RapidFuzz.

    Args:
        transcript_path: Path to the raw AMI transcript `.txt` file.
        target_quote: Semantic quote target proposed by the LLM or graph edge.
        threshold: Minimum `rapidfuzz.fuzz.partial_ratio` score (0-100), default 95.0.
        min_words: Minimum word count required by the evaluation harness (default 6).

    Returns:
        Exact un-normalized substring from a single speaker turn in `transcript_path`.

    Raises:
        CitationHallucinationError: If no speaker turn meets `threshold` or `min_words`.
    """
    path = Path(transcript_path)
    if not path.exists():
        raise CitationHallucinationError(f"Transcript file does not exist: {path}")

    norm_target, _ = normalize_with_char_map(target_quote)
    if not norm_target or len(norm_target) < 10:
        raise CitationHallucinationError(f"Target quote is too short for alignment: {target_quote!r}")

    doc = _get_cached_transcript(path)

    best_score: float = -1.0
    best_turn: Optional[SpeakerTurn] = None
    best_src_start: int = 0
    best_src_end: int = 0
    best_char_map: List[int] = []

    for turn in doc.turns:
        if turn.word_count < min_words:
            continue
        norm_turn, char_map = normalize_with_char_map(turn.text)
        if not norm_turn:
            continue

        alignment = fuzz.partial_ratio_alignment(norm_target, norm_turn)
        if alignment is None:
            continue

        score = float(alignment.score)
        # Index re-anchoring: strictly `>` preserves the earliest occurrence on ties
        if score > best_score:
            best_score = score
            best_turn = turn
            best_src_start = int(alignment.dest_start)
            best_src_end = int(alignment.dest_end)
            best_char_map = char_map

    if best_turn is None or best_score < threshold or not best_char_map:
        raise CitationHallucinationError(
            f"Rejected citation in {path.stem} (best RapidFuzz partial_ratio={best_score:.2f} < {threshold}): {target_quote!r}"
        )

    raw_start = best_char_map[max(0, min(best_src_start, len(best_char_map) - 1))]
    raw_end_char_idx = best_char_map[max(0, min(best_src_end - 1, len(best_char_map) - 1))] + 1

    exact_quote = _snap_to_word_boundaries(
        best_turn.text,
        raw_start,
        raw_end_char_idx,
        min_words=min_words,
    )

    if len(exact_quote.split()) < min_words:
        raise CitationHallucinationError(
            f"Extracted quote in {path.stem} has fewer than {min_words} words: {exact_quote!r}"
        )

    return exact_quote


def verify_citation_in_transcript(
    meeting_id: str,
    quote: str,
    transcripts_dir: Path | str = "transcripts",
) -> Tuple[bool, Optional[str]]:
    """Verify that `quote` has >= 6 words and matches a single speaker turn in `<meeting_id>.txt`."""
    tpath = Path(transcripts_dir) / f"{meeting_id}.txt"
    if not tpath.exists():
        return False, None
    if len(quote.split()) < 6:
        return False, None

    doc = _get_cached_transcript(tpath)
    norm_quote = re.sub(r"\s+", " ", quote.strip().lower())
    for turn in doc.turns:
        norm_turn = re.sub(r"\s+", " ", turn.text.strip().lower())
        if norm_quote in norm_turn:
            return True, turn.speaker_id
    return False, None


def audit_and_anchor_extracted_bundles(
    extracted_dir: Path | str = "data/extracted",
    transcripts_dir: Path | str = "transcripts",
    threshold: float = 95.0,
) -> Dict[str, int]:
    """Audit all extracted bundles through `locate_verbatim_quote`, anchoring every quote to exact raw text."""
    ext_base = Path(extracted_dir)
    trans_base = Path(transcripts_dir)

    total_checked = 0
    anchored_ok = 0
    rejected_count = 0

    for bundle_file in sorted(ext_base.glob("*.json")):
        bundle = TeamExtractionBundle.model_validate_json(bundle_file.read_text(encoding="utf-8"))

        valid_decisions = []
        for dec in bundle.decisions:
            total_checked += 1
            tpath = trans_base / f"{dec.meeting_id}.txt"
            try:
                dec.semantic_target_quote = locate_verbatim_quote(
                    tpath,
                    dec.semantic_target_quote,
                    threshold=threshold,
                )
                anchored_ok += 1
                valid_decisions.append(dec)
            except CitationHallucinationError:
                rejected_count += 1

        valid_actions = []
        for act in bundle.action_items:
            total_checked += 1
            tpath = trans_base / f"{act.meeting_id}.txt"
            try:
                act.semantic_target_quote = locate_verbatim_quote(
                    tpath,
                    act.semantic_target_quote,
                    threshold=threshold,
                )
                anchored_ok += 1
                valid_actions.append(act)
            except CitationHallucinationError:
                rejected_count += 1

        valid_disagreements = []
        for dis in bundle.disagreements:
            total_checked += 1
            tpath = trans_base / f"{dis.meeting_id}.txt"
            try:
                dis.semantic_target_quote = locate_verbatim_quote(
                    tpath,
                    dis.semantic_target_quote,
                    threshold=threshold,
                )
                anchored_ok += 1
                valid_disagreements.append(dis)
            except CitationHallucinationError:
                rejected_count += 1

        bundle.decisions = valid_decisions
        bundle.action_items = valid_actions
        bundle.disagreements = valid_disagreements
        bundle_file.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")

    return {
        "total_quotes_checked": total_checked,
        "verbatim_anchored_at_95_threshold": anchored_ok,
        "rejected_hallucinations": rejected_count,
    }


if __name__ == "__main__":
    stats = audit_and_anchor_extracted_bundles()
    print("Phase 3 Verbatim Citation Audit Complete:")
    print(json.dumps(stats, indent=2))
