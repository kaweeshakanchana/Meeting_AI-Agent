---
name: verbatim-citation
description: Implements deterministic string matching to extract exact quotes and prevent LLM hallucination, fulfilling the "tool-fit judgment" requirement.
---

# Verbatim Citation Extractor Workflow

Description: Implements deterministic string matching to extract exact quotes and prevent LLM hallucination, fulfilling the "tool-fit judgment" requirement.

## Implementation Specification

Create a Python module featuring `locate_verbatim_quote(transcript_path, target_quote, threshold=95)`:

1. **File I/O**: Read the raw, unedited text from `transcript_path`.
2. **Normalization**: Normalize both the source transcript and the `target_quote` (convert to lowercase, strip punctuation, and collapse whitespace).
3. **Fuzzy Alignment**: Utilize `rapidfuzz.fuzz.partial_ratio` (and `partial_ratio_alignment`) to compute the optimal alignment of the target within the source.
4. **Threshold Enforcement**: If the resulting score is `>= 95`, utilize the alignment indices to extract and return the exact, un-normalized substring from the original file (within a single speaker turn, at least 6 words).
5. **Rejection Logic**: If the score `< 95`, raise a custom `CitationHallucinationError`. This forces the pipeline to reject the LLM's claim and prevents hallucinated citations from entering the final JSON.
