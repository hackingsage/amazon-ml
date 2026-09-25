"""
inference.py — Output generation and validation.

Produces the two output TSV files:
  - output/matching_results.tsv
  - output/candidate_pairs.tsv

Also runs the validation checks matching utils/validate_submission.py's rules.
"""

import logging
from pathlib import Path
from typing import Dict, Set

import pandas as pd

logger = logging.getLogger(__name__)


def write_matching_results(
    predictions: Dict[str, Set[str]],
    output_path: str,
) -> None:
    """
    Write matching_results.tsv in the required format.
    
    Format:
      source1_entity_id<TAB>matched_entity_ids
      S1-00001<TAB>S2-00047,S2-00193,S3-00812
      S1-00002<TAB>S3-00004
      S1-00003<TAB>
    
    Rules:
      - One row per Source 1 entity
      - matched_entity_ids: comma-separated, no quoting
      - Empty string for singletons (no match)
      - No duplicate IDs within a row
      - Only S2/S3 IDs
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    rows = []
    for s1_id in sorted(predictions.keys()):
        matched_ids = sorted(predictions[s1_id])
        matched_str = ",".join(matched_ids)
        rows.append({"source1_entity_id": s1_id, "matched_entity_ids": matched_str})
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep="\t", index=False)
    
    logger.info(f"Wrote matching_results.tsv: {len(df)} rows -> {output_path}")
    
    # Quick stats
    n_matched = sum(1 for _, row in df.iterrows() if row["matched_entity_ids"])
    n_singleton = len(df) - n_matched
    logger.info(f"  Matched entities: {n_matched}, Singletons: {n_singleton}")


def write_candidate_pairs(
    candidates: Dict[str, Set[str]],
    output_path: str,
) -> None:
    """
    Write candidate_pairs.tsv in the required format.
    
    Format:
      source1_entity_id<TAB>candidate_entity_ids
      S1-00001<TAB>S2-00047,S2-00193,S3-00812,S3-00999
    
    Same rules as matching_results.tsv.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    rows = []
    for s1_id in sorted(candidates.keys()):
        cand_ids = sorted(candidates[s1_id])
        cand_str = ",".join(cand_ids)
        rows.append({"source1_entity_id": s1_id, "candidate_entity_ids": cand_str})
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep="\t", index=False)
    
    logger.info(f"Wrote candidate_pairs.tsv: {len(df)} rows -> {output_path}")


def validate_submission(
    matching_path: str,
    candidate_path: str,
    test_s1_ids: Set[str],
    test_s23_ids: Set[str],
) -> bool:
    """
    Validate the output files against the submission rules.
    
    Mirrors the checks in utils/validate_submission.py:
    1. Both files exist and are readable
    2. Correct column names
    3. Every test S1 entity has exactly one row
    4. No duplicate S1 rows
    5. All matched/candidate IDs are valid S2/S3 IDs from the test set
    6. No duplicate IDs within a single row's ID list
    7. Every matched ID appears in the candidate set (matched ⊆ candidates)
    8. No S1 self-matches
    
    Returns:
        True if all checks pass, False otherwise.
    """
    logger.info("=" * 60)
    logger.info("VALIDATION")
    logger.info("=" * 60)
    
    issues = []
    
    # --- Check 1: Files exist ---
    if not Path(matching_path).exists():
        issues.append(f"matching_results.tsv not found: {matching_path}")
    if not Path(candidate_path).exists():
        issues.append(f"candidate_pairs.tsv not found: {candidate_path}")
    
    if issues:
        for issue in issues:
            logger.error(f"  FAIL: {issue}")
        return False
    
    # --- Load files ---
    try:
        matching_df = pd.read_csv(matching_path, sep="\t", dtype=str, keep_default_na=False)
    except Exception as e:
        issues.append(f"Cannot read matching_results.tsv: {e}")
        for issue in issues:
            logger.error(f"  FAIL: {issue}")
        return False
    
    try:
        candidate_df = pd.read_csv(candidate_path, sep="\t", dtype=str, keep_default_na=False)
    except Exception as e:
        issues.append(f"Cannot read candidate_pairs.tsv: {e}")
        for issue in issues:
            logger.error(f"  FAIL: {issue}")
        return False
    
    # --- Check 2: Correct column names ---
    expected_matching_cols = {"source1_entity_id", "matched_entity_ids"}
    expected_candidate_cols = {"source1_entity_id", "candidate_entity_ids"}
    
    if set(matching_df.columns) != expected_matching_cols:
        issues.append(f"matching_results.tsv columns should be {expected_matching_cols}, got {set(matching_df.columns)}")
    if set(candidate_df.columns) != expected_candidate_cols:
        issues.append(f"candidate_pairs.tsv columns should be {expected_candidate_cols}, got {set(candidate_df.columns)}")
    
    if issues:
        for issue in issues:
            logger.error(f"  FAIL: {issue}")
        return False
    
    # --- Check 3 & 4: Every test S1 entity has exactly one row, no duplicates ---
    matching_s1_ids = matching_df["source1_entity_id"].tolist()
    candidate_s1_ids = candidate_df["source1_entity_id"].tolist()
    
    # Duplicates
    matching_dupes = [x for x in matching_s1_ids if matching_s1_ids.count(x) > 1]
    if matching_dupes:
        issues.append(f"Duplicate S1 rows in matching_results.tsv: {set(matching_dupes)}")
    
    candidate_dupes = [x for x in candidate_s1_ids if candidate_s1_ids.count(x) > 1]
    if candidate_dupes:
        issues.append(f"Duplicate S1 rows in candidate_pairs.tsv: {set(candidate_dupes)}")
    
    # Missing S1 entities
    matching_s1_set = set(matching_s1_ids)
    candidate_s1_set = set(candidate_s1_ids)
    
    missing_matching = test_s1_ids - matching_s1_set
    if missing_matching:
        issues.append(f"Missing S1 entities in matching_results.tsv: {len(missing_matching)} entities")
    
    missing_candidate = test_s1_ids - candidate_s1_set
    if missing_candidate:
        issues.append(f"Missing S1 entities in candidate_pairs.tsv: {len(missing_candidate)} entities")
    
    extra_matching = matching_s1_set - test_s1_ids
    if extra_matching:
        issues.append(f"Extra S1 entities in matching_results.tsv (not in test): {len(extra_matching)} entities")
    
    # --- Check 5: Valid S2/S3 IDs ---
    for idx, row in matching_df.iterrows():
        id_str = str(row.get("matched_entity_ids", ""))
        if not id_str:
            continue
        ids = id_str.split(",")
        for eid in ids:
            eid = eid.strip()
            if not eid:
                continue
            if eid.startswith("S1-"):
                issues.append(f"S1 self-match in matching_results.tsv row {idx}: {eid}")
            if eid not in test_s23_ids:
                issues.append(f"Unknown ID in matching_results.tsv row {idx}: {eid}")
    
    for idx, row in candidate_df.iterrows():
        id_str = str(row.get("candidate_entity_ids", ""))
        if not id_str:
            continue
        ids = id_str.split(",")
        for eid in ids:
            eid = eid.strip()
            if not eid:
                continue
            if eid.startswith("S1-"):
                issues.append(f"S1 self-match in candidate_pairs.tsv row {idx}: {eid}")
            if eid not in test_s23_ids:
                issues.append(f"Unknown ID in candidate_pairs.tsv row {idx}: {eid}")
    
    # --- Check 6: No duplicate IDs within a row ---
    for idx, row in matching_df.iterrows():
        id_str = str(row.get("matched_entity_ids", ""))
        if not id_str:
            continue
        ids = [x.strip() for x in id_str.split(",") if x.strip()]
        if len(ids) != len(set(ids)):
            issues.append(f"Duplicate IDs in matching_results.tsv row {idx}")
    
    for idx, row in candidate_df.iterrows():
        id_str = str(row.get("candidate_entity_ids", ""))
        if not id_str:
            continue
        ids = [x.strip() for x in id_str.split(",") if x.strip()]
        if len(ids) != len(set(ids)):
            issues.append(f"Duplicate IDs in candidate_pairs.tsv row {idx}")
    
    # --- Check 7: matched IDs ⊆ candidate IDs ---
    # Build candidate lookup
    candidate_lookup: Dict[str, Set[str]] = {}
    for _, row in candidate_df.iterrows():
        s1_id = row["source1_entity_id"]
        id_str = str(row.get("candidate_entity_ids", ""))
        cands = set(x.strip() for x in id_str.split(",") if x.strip()) if id_str else set()
        candidate_lookup[s1_id] = cands
    
    for _, row in matching_df.iterrows():
        s1_id = row["source1_entity_id"]
        id_str = str(row.get("matched_entity_ids", ""))
        matches = set(x.strip() for x in id_str.split(",") if x.strip()) if id_str else set()
        cands = candidate_lookup.get(s1_id, set())
        not_in_cands = matches - cands
        if not_in_cands:
            issues.append(f"Matched IDs not in candidates for {s1_id}: {not_in_cands}")
    
    # --- Report ---
    if issues:
        logger.error(f"\nVALIDATION FAILED - {len(issues)} issue(s):")
        for i, issue in enumerate(issues[:20], 1):  # cap at 20 displayed
            logger.error(f"  {i}. {issue}")
        if len(issues) > 20:
            logger.error(f"  ... and {len(issues) - 20} more issues")
        return False
    
    logger.info("  [PASS] All validation checks PASSED")
    return True
