"""
blocking.py — Candidate generation via union of token-based + TF-IDF n-gram blocking.

Produces candidate pairs: for each S1 entity, a set of S2/S3 entity_ids that are
plausible matches. The union of two complementary strategies maximizes recall ceiling.

Strategy A: Token-based blocking
  - Blocking keys = {country, name_token} pairs (after suffix stripping).
  - Two records share a block if they share any blocking key.
  - Fast, high precision, but misses fuzzy/abbreviated matches.

Strategy B: TF-IDF character n-gram blocking
  - Build TF-IDF on character 3-grams of concatenated name+address+country.
  - For each S1 entity, retrieve top-K most similar S2/S3 records by cosine.
  - Catches fuzzy matches, transliterations, abbreviations.

Final candidate set = A ∪ B.
"""

import gc
import logging
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from .normalization import (
    get_combined_text,
    get_name_tokens_for_blocking,
    normalize_country,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy A: Token-based blocking
# ---------------------------------------------------------------------------

def _build_inverted_index(
    df: pd.DataFrame,
    id_col: str = "entity_id",
    name_col: str = "business_name",
    country_col: str = "country",
) -> Dict[str, Set[str]]:
    """
    Build an inverted index: blocking_key → set of entity_ids.
    Blocking keys are (country, name_token) tuples stringified.
    """
    index: Dict[str, Set[str]] = defaultdict(set)
    for _, row in df.iterrows():
        eid = row[id_col]
        country = normalize_country(row.get(country_col, ""))
        name_tokens = get_name_tokens_for_blocking(str(row.get(name_col, "")))
        for token in name_tokens:
            # Key = country + token (compound key for selectivity)
            key = f"{country}|{token}"
            index[key].add(eid)
            # Also add token-only key (catches cross-country edge cases)
            key_no_country = f"*|{token}"
            index[key_no_country].add(eid)
    return index


def token_blocking(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    max_block_size: int = 500,
) -> Dict[str, Set[str]]:
    """
    Token-based blocking: for each S1 entity, find S2/S3 entities sharing
    at least one blocking key.
    
    Args:
        s1_df: Source 1 dataframe.
        s23_df: Combined Source 2 + Source 3 dataframe.
        max_block_size: Skip blocks larger than this (too common tokens).
    
    Returns:
        Dict mapping S1 entity_id → set of candidate S2/S3 entity_ids.
    """
    logger.info("Building token-based blocking index for S2/S3...")
    s23_index = _build_inverted_index(s23_df)
    
    # Filter out overly large blocks (stopword-like tokens)
    s23_index = {k: v for k, v in s23_index.items() if len(v) <= max_block_size}
    logger.info(f"  S2/S3 index: {len(s23_index)} blocking keys (after filtering blocks > {max_block_size})")
    
    candidates: Dict[str, Set[str]] = {}
    for _, row in s1_df.iterrows():
        eid = row["entity_id"]
        country = normalize_country(row.get("country", ""))
        name_tokens = get_name_tokens_for_blocking(str(row.get("business_name", "")))
        
        cands: Set[str] = set()
        for token in name_tokens:
            # Check country-specific key first
            key = f"{country}|{token}"
            if key in s23_index:
                cands.update(s23_index[key])
            # Also check wildcard key
            key_no_country = f"*|{token}"
            if key_no_country in s23_index:
                cands.update(s23_index[key_no_country])
        
        candidates[eid] = cands
    
    total_pairs = sum(len(v) for v in candidates.values())
    logger.info(f"  Token blocking: {len(candidates)} S1 entities → {total_pairs} total candidate pairs")
    return candidates


# ---------------------------------------------------------------------------
# Strategy B: TF-IDF character n-gram blocking
# ---------------------------------------------------------------------------

def tfidf_blocking(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    top_k: int = 50,
    ngram_range: Tuple[int, int] = (3, 4),
    min_df: int = 3,
    max_df: float = 0.8,
    max_features: int = 400000,
    batch_size: int = 1000,
) -> Dict[str, Set[str]]:
    """
    TF-IDF character n-gram blocking: for each S1 entity, retrieve the top-K
    most similar S2/S3 entities by cosine similarity on character n-grams.
    
    Calibrated for a ~28-30 GB RAM budget (utilizing available 32 GB system memory):
      - Uses (3, 4)-grams with max_features=400,000 for maximum fine-grained vocabulary
      - min_df=3 captures low-frequency entity terms and domain-specific words
      - top_k=50 boosts candidate recall ceiling
      - batch_size=1000 maximizes multi-threaded matrix multiplication throughput
      - Direct sparse CSR slicing avoids allocating dense arrays during top-k selection
      - Explicit garbage collection frees intermediate matrices
    
    Args:
        s1_df: Source 1 dataframe.
        s23_df: Combined Source 2 + Source 3 dataframe.
        top_k: Number of top candidates to retrieve per S1 entity.
        ngram_range: Character n-gram range for TF-IDF.
        min_df: Minimum document frequency for TF-IDF terms.
        max_df: Maximum document frequency fraction for TF-IDF terms.
        max_features: Maximum vocabulary size to prevent OOM.
        batch_size: Batch size for cosine similarity matrix multiplication.
    
    Returns:
        Dict mapping S1 entity_id → set of candidate S2/S3 entity_ids.
    """
    logger.info("Building TF-IDF character n-gram blocking...")
    
    # Build combined text for all records
    s1_texts = []
    s1_ids = []
    for _, row in s1_df.iterrows():
        s1_ids.append(row["entity_id"])
        s1_texts.append(get_combined_text(
            str(row.get("business_name", "")),
            str(row.get("business_address", "")),
            str(row.get("country", "")),
        ))
    
    s23_texts = []
    s23_ids = []
    for _, row in s23_df.iterrows():
        s23_ids.append(row["entity_id"])
        s23_texts.append(get_combined_text(
            str(row.get("business_name", "")),
            str(row.get("business_address", "")),
            str(row.get("country", "")),
        ))
    
    # Fit TF-IDF on all texts jointly for consistent vocabulary
    all_texts = s1_texts + s23_texts
    
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    
    logger.info(
        f"  Fitting TF-IDF on {len(all_texts)} documents "
        f"(ngram_range={ngram_range}, max_features={max_features}, min_df={min_df})..."
    )
    all_tfidf = vectorizer.fit_transform(all_texts)
    
    del all_texts
    gc.collect()
    
    s1_tfidf = all_tfidf[:len(s1_texts)]
    s23_tfidf = all_tfidf[len(s1_texts):]
    
    logger.info(f"  TF-IDF matrix: {all_tfidf.shape[0]} docs × {all_tfidf.shape[1]} features")
    
    # Compute top-K for each S1 entity using sparse dot product
    # We do this in batches to manage memory
    candidates: Dict[str, Set[str]] = {}
    
    s23_tfidf_T = s23_tfidf.T.tocsc()  # transpose for fast column access
    
    for batch_start in range(0, len(s1_ids), batch_size):
        batch_end = min(batch_start + batch_size, len(s1_ids))
        batch_tfidf = s1_tfidf[batch_start:batch_end]
        
        # Cosine similarity = dot product (vectors are already L2-normalized by default norm='l2')
        sim_matrix = batch_tfidf.dot(s23_tfidf_T)
        
        # Fast direct sparse CSR extraction (avoids allocating 1M+ dense float arrays)
        indptr = sim_matrix.indptr
        indices = sim_matrix.indices
        data = sim_matrix.data
        
        for i in range(batch_end - batch_start):
            s1_id = s1_ids[batch_start + i]
            row_start = indptr[i]
            row_end = indptr[i + 1]
            row_len = row_end - row_start
            
            if row_len == 0:
                candidates[s1_id] = set()
                continue
            
            row_data = data[row_start:row_end]
            row_indices = indices[row_start:row_end]
            
            if row_len <= top_k:
                top_indices = row_indices[row_data > 0]
            else:
                part = np.argpartition(row_data, -top_k)[-top_k:]
                part = part[row_data[part] > 0]
                top_indices = row_indices[part]
            
            candidates[s1_id] = {s23_ids[idx] for idx in top_indices}
    
    del all_tfidf, s1_tfidf, s23_tfidf, s23_tfidf_T
    gc.collect()
    
    total_pairs = sum(len(v) for v in candidates.values())
    logger.info(f"  TF-IDF blocking: {len(candidates)} S1 entities → {total_pairs} total candidate pairs (top_k={top_k})")
    return candidates


# ---------------------------------------------------------------------------
# Union blocking
# ---------------------------------------------------------------------------

def union_blocking(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    top_k: int = 50,
    max_block_size: int = 1000,
    tfidf_max_features: int = 400000,
    tfidf_min_df: int = 3,
    tfidf_ngram_range: Tuple[int, int] = (3, 4),
) -> Dict[str, Set[str]]:
    """
    Union of token-based + TF-IDF blocking. Maximizes recall ceiling.
    
    Args:
        s1_df: Source 1 dataframe.
        s23_df: Combined Source 2 + Source 3 dataframe.
        top_k: Top-K for TF-IDF blocking.
        max_block_size: Max block size for token blocking.
        tfidf_max_features: Cap on TF-IDF features to fit within RAM budget.
        tfidf_min_df: Min doc frequency for TF-IDF terms.
        tfidf_ngram_range: N-gram range for TF-IDF.
    
    Returns:
        Dict mapping S1 entity_id → set of candidate S2/S3 entity_ids.
    """
    logger.info("=" * 60)
    logger.info("BLOCKING STAGE: Union of Token + TF-IDF n-gram")
    logger.info("=" * 60)
    
    token_cands = token_blocking(s1_df, s23_df, max_block_size=max_block_size)
    tfidf_cands = tfidf_blocking(
        s1_df,
        s23_df,
        top_k=top_k,
        ngram_range=tfidf_ngram_range,
        max_features=tfidf_max_features,
        min_df=tfidf_min_df,
    )
    
    # Union
    all_s1_ids = set(s1_df["entity_id"].tolist())
    candidates: Dict[str, Set[str]] = {}
    
    for s1_id in all_s1_ids:
        cands = set()
        if s1_id in token_cands:
            cands.update(token_cands[s1_id])
        if s1_id in tfidf_cands:
            cands.update(tfidf_cands[s1_id])
        candidates[s1_id] = cands
    
    total_pairs = sum(len(v) for v in candidates.values())
    avg_cands = total_pairs / max(len(candidates), 1)
    logger.info(f"\nUnion blocking summary:")
    logger.info(f"  S1 entities: {len(candidates)}")
    logger.info(f"  Total candidate pairs: {total_pairs}")
    logger.info(f"  Avg candidates per S1 entity: {avg_cands:.1f}")
    
    # Token-only stats
    token_total = sum(len(v) for v in token_cands.values())
    tfidf_total = sum(len(v) for v in tfidf_cands.values())
    logger.info(f"  Token-only pairs: {token_total}")
    logger.info(f"  TF-IDF-only pairs: {tfidf_total}")
    logger.info(f"  Union pairs: {total_pairs}")
    
    return candidates


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_blocking(
    candidates: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
    total_s23_entities: int,
) -> Dict[str, float]:
    """
    Evaluate blocking quality on a validation split.
    
    Args:
        candidates: S1 entity_id → set of candidate S2/S3 entity_ids.
        ground_truth: S1 entity_id → set of true matching S2/S3 entity_ids.
        total_s23_entities: Total number of S2/S3 entities (for reduction ratio).
    
    Returns:
        Dict with 'recall_ceiling', 'reduction_ratio', 'avg_candidates'.
    """
    total_true = 0
    total_found = 0
    total_candidate_pairs = 0
    total_possible_pairs = 0
    
    for s1_id, true_matches in ground_truth.items():
        if not true_matches:
            continue  # singletons don't affect recall ceiling
        
        cands = candidates.get(s1_id, set())
        found = true_matches & cands
        
        total_true += len(true_matches)
        total_found += len(found)
        total_candidate_pairs += len(cands)
        total_possible_pairs += total_s23_entities
    
    # Also count candidate pairs for singletons
    for s1_id in candidates:
        if s1_id not in ground_truth or not ground_truth[s1_id]:
            total_candidate_pairs += len(candidates[s1_id])
            total_possible_pairs += total_s23_entities
    
    recall_ceiling = total_found / max(total_true, 1)
    reduction_ratio = 1.0 - (total_candidate_pairs / max(total_possible_pairs, 1))
    avg_candidates = total_candidate_pairs / max(len(candidates), 1)
    
    return {
        "recall_ceiling": recall_ceiling,
        "reduction_ratio": reduction_ratio,
        "avg_candidates": avg_candidates,
        "total_true_matches": total_true,
        "true_matches_found_in_candidates": total_found,
        "true_matches_missed": total_true - total_found,
        "total_candidate_pairs": total_candidate_pairs,
    }
