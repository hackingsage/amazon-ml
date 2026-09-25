"""
features.py — Pairwise similarity feature engineering for candidate pairs.

Computes ~30 features for each (S1 entity, S2/S3 entity) pair covering:
  - Name similarity (multiple metrics)
  - Address similarity (multiple metrics)
  - Country match
  - Token overlap statistics
  - Length ratios
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .normalization import (
    normalize_address,
    normalize_business_name,
    normalize_country,
    normalize_text,
    tokenize,
    strip_legal_suffixes,
    normalize_legal_suffixes,
    normalize_address_abbrevs,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# String similarity functions (no external dependency on jellyfish etc.)
# ---------------------------------------------------------------------------

def levenshtein_distance(s1: str, s2: str) -> int:
    """Standard Levenshtein edit distance via dynamic programming."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    
    return prev_row[-1]


def levenshtein_ratio(s1: str, s2: str) -> float:
    """Levenshtein similarity ratio: 1 - (distance / max_len)."""
    if not s1 and not s2:
        return 1.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return 1.0 - levenshtein_distance(s1, s2) / max_len


def jaro_similarity(s1: str, s2: str) -> float:
    """Jaro similarity between two strings."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    
    len1, len2 = len(s1), len(s2)
    max_dist = max(len1, len2) // 2 - 1
    if max_dist < 0:
        max_dist = 0
    
    s1_matches = [False] * len1
    s2_matches = [False] * len2
    
    matches = 0
    transpositions = 0
    
    for i in range(len1):
        start = max(0, i - max_dist)
        end = min(i + max_dist + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break
    
    if matches == 0:
        return 0.0
    
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    
    return (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3


def jaro_winkler_similarity(s1: str, s2: str, p: float = 0.1) -> float:
    """Jaro-Winkler similarity (boosts score for common prefixes)."""
    jaro = jaro_similarity(s1, s2)
    
    # Find common prefix length (up to 4)
    prefix_len = 0
    for i in range(min(len(s1), len(s2), 4)):
        if s1[i] == s2[i]:
            prefix_len += 1
        else:
            break
    
    return jaro + prefix_len * p * (1 - jaro)


def jaccard_similarity(set1: set, set2: set) -> float:
    """Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def overlap_coefficient(set1: set, set2: set) -> float:
    """Overlap coefficient: |intersection| / min(|set1|, |set2|)."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    min_size = min(len(set1), len(set2))
    return intersection / min_size if min_size > 0 else 0.0


def containment_similarity(s1: str, s2: str) -> float:
    """Check if one string contains the other (for abbreviation matching)."""
    if not s1 or not s2:
        return 0.0
    if s1 in s2 or s2 in s1:
        return 1.0
    return 0.0


def char_ngram_jaccard(s1: str, s2: str, n: int = 3) -> float:
    """Jaccard similarity on character n-grams."""
    if not s1 or not s2:
        return 0.0
    ngrams1 = {s1[i:i+n] for i in range(len(s1) - n + 1)} if len(s1) >= n else {s1}
    ngrams2 = {s2[i:i+n] for i in range(len(s2) - n + 1)} if len(s2) >= n else {s2}
    return jaccard_similarity(ngrams1, ngrams2)


# ---------------------------------------------------------------------------
# Feature computation for a single pair
# ---------------------------------------------------------------------------

def compute_pair_features(
    s1_row: dict,
    s23_row: dict,
) -> Dict[str, float]:
    """
    Compute all pairwise similarity features for one (S1, S2/S3) pair.
    
    Args:
        s1_row: Dict with keys: business_name, business_address, country.
        s23_row: Dict with keys: business_name, business_address, country.
    
    Returns:
        Dict of feature_name → float_value.
    """
    features = {}
    
    # --- Raw fields ---
    name1_raw = str(s1_row.get("business_name", ""))
    name2_raw = str(s23_row.get("business_name", ""))
    addr1_raw = str(s1_row.get("business_address", ""))
    addr2_raw = str(s23_row.get("business_address", ""))
    country1 = normalize_country(s1_row.get("country", ""))
    country2 = normalize_country(s23_row.get("country", ""))
    
    # --- Normalized fields ---
    name1_norm = normalize_business_name(name1_raw, keep_suffixes=True)
    name2_norm = normalize_business_name(name2_raw, keep_suffixes=True)
    name1_no_suffix = normalize_business_name(name1_raw, keep_suffixes=False)
    name2_no_suffix = normalize_business_name(name2_raw, keep_suffixes=False)
    addr1_norm = normalize_address(addr1_raw)
    addr2_norm = normalize_address(addr2_raw)
    
    # --- Name tokens ---
    name1_tokens = set(name1_norm.split()) if name1_norm else set()
    name2_tokens = set(name2_norm.split()) if name2_norm else set()
    name1_ns_tokens = set(name1_no_suffix.split()) if name1_no_suffix else set()
    name2_ns_tokens = set(name2_no_suffix.split()) if name2_no_suffix else set()
    
    # --- Address tokens ---
    addr1_tokens = set(addr1_norm.split()) if addr1_norm else set()
    addr2_tokens = set(addr2_norm.split()) if addr2_norm else set()
    
    # ===================================================================
    # NAME FEATURES
    # ===================================================================
    
    # String-level similarity on full normalized name
    features["name_jaro_winkler"] = jaro_winkler_similarity(name1_norm, name2_norm)
    features["name_levenshtein_ratio"] = levenshtein_ratio(name1_norm, name2_norm)
    features["name_char_3gram_jaccard"] = char_ngram_jaccard(name1_norm, name2_norm, n=3)
    features["name_char_4gram_jaccard"] = char_ngram_jaccard(name1_norm, name2_norm, n=4)
    
    # String-level similarity on name without legal suffixes
    features["name_ns_jaro_winkler"] = jaro_winkler_similarity(name1_no_suffix, name2_no_suffix)
    features["name_ns_levenshtein_ratio"] = levenshtein_ratio(name1_no_suffix, name2_no_suffix)
    
    # Exact match flags
    features["name_exact_match"] = 1.0 if name1_norm == name2_norm and name1_norm else 0.0
    features["name_ns_exact_match"] = 1.0 if name1_no_suffix == name2_no_suffix and name1_no_suffix else 0.0
    
    # Token-level similarity
    features["name_token_jaccard"] = jaccard_similarity(name1_tokens, name2_tokens)
    features["name_token_overlap"] = overlap_coefficient(name1_tokens, name2_tokens)
    features["name_ns_token_jaccard"] = jaccard_similarity(name1_ns_tokens, name2_ns_tokens)
    features["name_ns_token_overlap"] = overlap_coefficient(name1_ns_tokens, name2_ns_tokens)
    
    # Token counts
    features["name_shared_token_count"] = float(len(name1_tokens & name2_tokens))
    features["name_ns_shared_token_count"] = float(len(name1_ns_tokens & name2_ns_tokens))
    
    # Containment
    features["name_containment"] = containment_similarity(name1_no_suffix, name2_no_suffix)
    
    # Length ratio
    len1 = len(name1_norm) if name1_norm else 0
    len2 = len(name2_norm) if name2_norm else 0
    features["name_length_ratio"] = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 1.0
    
    # Token count ratio
    tc1 = len(name1_ns_tokens)
    tc2 = len(name2_ns_tokens)
    features["name_token_count_ratio"] = min(tc1, tc2) / max(tc1, tc2) if max(tc1, tc2) > 0 else 1.0
    
    # ===================================================================
    # ADDRESS FEATURES
    # ===================================================================
    
    # String-level similarity
    features["addr_jaro_winkler"] = jaro_winkler_similarity(addr1_norm, addr2_norm)
    features["addr_levenshtein_ratio"] = levenshtein_ratio(addr1_norm, addr2_norm)
    features["addr_char_3gram_jaccard"] = char_ngram_jaccard(addr1_norm, addr2_norm, n=3)
    
    # Exact match
    features["addr_exact_match"] = 1.0 if addr1_norm == addr2_norm and addr1_norm else 0.0
    
    # Token-level similarity
    features["addr_token_jaccard"] = jaccard_similarity(addr1_tokens, addr2_tokens)
    features["addr_token_overlap"] = overlap_coefficient(addr1_tokens, addr2_tokens)
    features["addr_shared_token_count"] = float(len(addr1_tokens & addr2_tokens))
    
    # Length ratio
    al1 = len(addr1_norm) if addr1_norm else 0
    al2 = len(addr2_norm) if addr2_norm else 0
    features["addr_length_ratio"] = min(al1, al2) / max(al1, al2) if max(al1, al2) > 0 else 1.0
    
    # ===================================================================
    # COUNTRY FEATURES
    # ===================================================================
    features["country_exact_match"] = 1.0 if country1 == country2 and country1 else 0.0
    features["country_either_missing"] = 1.0 if (not country1 or not country2) else 0.0
    
    # ===================================================================
    # COMBINED FEATURES
    # ===================================================================
    
    # Combined name+address text similarity
    combined1 = f"{name1_norm} {addr1_norm}".strip()
    combined2 = f"{name2_norm} {addr2_norm}".strip()
    features["combined_jaro_winkler"] = jaro_winkler_similarity(combined1, combined2)
    features["combined_char_3gram_jaccard"] = char_ngram_jaccard(combined1, combined2, n=3)
    
    # Combined token overlap
    all_tokens1 = name1_tokens | addr1_tokens
    all_tokens2 = name2_tokens | addr2_tokens
    features["combined_token_jaccard"] = jaccard_similarity(all_tokens1, all_tokens2)
    
    # Source indicator (S2 vs S3) — the noise patterns may differ
    s23_id = str(s23_row.get("entity_id", ""))
    features["is_source3"] = 1.0 if s23_id.startswith("S3-") else 0.0
    
    return features


# ---------------------------------------------------------------------------
# Batch feature computation
# ---------------------------------------------------------------------------

FEATURE_NAMES: Optional[List[str]] = None  # set after first call


def compute_features_for_pairs(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    candidate_pairs: Dict[str, set],
) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """
    Compute features for all candidate pairs.
    
    Args:
        s1_df: Source 1 dataframe.
        s23_df: Combined Source 2 + Source 3 dataframe.
        candidate_pairs: S1 entity_id → set of candidate S2/S3 entity_ids.
    
    Returns:
        (feature_matrix_df, pair_ids) where:
          - feature_matrix_df: DataFrame with one row per pair, columns = feature names
          - pair_ids: list of (s1_id, s23_id) tuples in same order
    """
    global FEATURE_NAMES
    
    logger.info("Computing pairwise features for candidate pairs...")
    
    # Index S1 and S23 by entity_id for fast lookup
    s1_lookup = {row["entity_id"]: row.to_dict() for _, row in s1_df.iterrows()}
    s23_lookup = {row["entity_id"]: row.to_dict() for _, row in s23_df.iterrows()}
    
    all_features = []
    pair_ids = []
    
    total_pairs = sum(len(v) for v in candidate_pairs.items())
    logger.info(f"  Total pairs to compute features for: {total_pairs}")
    
    processed = 0
    for s1_id, cand_ids in candidate_pairs.items():
        s1_row = s1_lookup.get(s1_id)
        if s1_row is None:
            continue
        
        for s23_id in cand_ids:
            s23_row = s23_lookup.get(s23_id)
            if s23_row is None:
                continue
            
            feats = compute_pair_features(s1_row, s23_row)
            all_features.append(feats)
            pair_ids.append((s1_id, s23_id))
            
            processed += 1
            if processed % 10000 == 0:
                logger.info(f"  Processed {processed}/{total_pairs} pairs...")
    
    if all_features:
        feature_df = pd.DataFrame(all_features)
        FEATURE_NAMES = list(feature_df.columns)
        logger.info(f"  Feature matrix shape: {feature_df.shape}")
        logger.info(f"  Features: {FEATURE_NAMES}")
    else:
        feature_df = pd.DataFrame()
        logger.warning("  No features computed — empty candidate set!")
    
    return feature_df, pair_ids
