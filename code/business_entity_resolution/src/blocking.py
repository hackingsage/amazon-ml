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
import time
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
    ids = df[id_col].tolist()
    countries = df[country_col].tolist() if country_col in df.columns else [""] * len(df)
    names = df[name_col].tolist() if name_col in df.columns else [""] * len(df)
    for eid, country_raw, name_raw in zip(ids, countries, names):
        country = normalize_country(country_raw)
        name_tokens = get_name_tokens_for_blocking(str(name_raw))
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
    s1_ids = s1_df["entity_id"].tolist()
    s1_countries = s1_df["country"].tolist() if "country" in s1_df.columns else [""] * len(s1_df)
    s1_names = s1_df["business_name"].tolist() if "business_name" in s1_df.columns else [""] * len(s1_df)

    for eid, country_raw, name_raw in zip(s1_ids, s1_countries, s1_names):
        country = normalize_country(country_raw)
        name_tokens = get_name_tokens_for_blocking(str(name_raw))
        
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
    max_terms_per_doc: int = 30,
    max_posting_list_size: int = 2000,
    max_candidate_pool_size: int = 3000,
) -> Dict[str, Set[str]]:
    """
    TF-IDF character n-gram blocking: for each S1 entity, retrieve the top-K
    most similar S2/S3 entities by cosine similarity on character n-grams.

    SPEED NOTE: exact brute-force cosine against the full S2/S3 corpus is
    O(n_s1 * n_s23) sparse work. At S2/S3 corpus sizes in the tens of
    millions this is simply too slow (hundreds of hours), no matter how well
    the matmul itself is parallelized. So instead of comparing every S1 row
    against every S23 row, we use an inverted index over TF-IDF terms to
    pre-filter each query down to a small candidate set (the S23 docs that
    actually share a distinctive n-gram with it), and only run exact cosine
    within that much smaller set. This is the same "reduce before you compare"
    idea as Strategy A (token blocking), applied to the TF-IDF vocabulary:
      1. For each S23 doc, keep only its top `max_terms_per_doc` highest-
         weighted (most distinctive) terms, and build term -> doc postings.
      2. Terms that appear in more than `max_posting_list_size` docs are
         dropped from the index (they're too common to be selective, same
         idea as `max_block_size` in token_blocking) — this bounds the
         candidate set size per query regardless of corpus size.
      3. For each S1 doc, look up postings for its own top terms, union
         them into a small candidate pool, then compute exact cosine
         similarity only against that pool and keep the top-K.
    This turns the search from O(n_s1 * n_s23) into roughly
    O(n_s1 * max_terms_per_doc * max_posting_list_size), independent of
    total corpus size.

    Args:
        s1_df: Source 1 dataframe.
        s23_df: Combined Source 2 + Source 3 dataframe.
        top_k: Number of top candidates to retrieve per S1 entity.
        ngram_range: Character n-gram range for TF-IDF.
        min_df: Minimum document frequency for TF-IDF terms.
        max_df: Maximum document frequency fraction for TF-IDF terms.
        max_features: Maximum vocabulary size to prevent OOM.
        batch_size: Batch size for progress logging / chunking.
        max_terms_per_doc: How many of each doc's highest-weighted terms to
            index/query with. Higher = better recall, slower & more memory.
        max_posting_list_size: Drop terms whose posting list (number of docs
            containing them) exceeds this. Bounds worst-case candidate pool
            size per query and keeps the index selective.
        max_candidate_pool_size: Hard cap on how many S23 docs are gathered
            per query before exact cosine is computed. Terms are consumed
            rarest-first (most discriminative first), so hitting this cap
            still keeps the most useful candidates. This is what makes
            per-query cost bounded and independent of corpus size, even if
            posting-list statistics are less selective than expected.

    Returns:
        Dict mapping S1 entity_id → set of candidate S2/S3 entity_ids.
    """
    logger.info("Building TF-IDF character n-gram blocking...")
    
    # Build combined text for all records
    def _cols(df: pd.DataFrame) -> Tuple[list, list, list, list]:
        n = len(df)
        ids = df["entity_id"].tolist()
        names = df["business_name"].tolist() if "business_name" in df.columns else [""] * n
        addrs = df["business_address"].tolist() if "business_address" in df.columns else [""] * n
        countries = df["country"].tolist() if "country" in df.columns else [""] * n
        return ids, names, addrs, countries

    s1_ids, s1_names, s1_addrs, s1_countries = _cols(s1_df)
    s1_texts = [
        get_combined_text(str(name), str(addr), str(country))
        for name, addr, country in zip(s1_names, s1_addrs, s1_countries)
    ]

    s23_ids, s23_names, s23_addrs, s23_countries = _cols(s23_df)
    s23_texts = [
        get_combined_text(str(name), str(addr), str(country))
        for name, addr, country in zip(s23_names, s23_addrs, s23_countries)
    ]
    
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
    vectorize_t0 = time.time()
    all_tfidf = vectorizer.fit_transform(all_texts)
    vectorize_elapsed = time.time() - vectorize_t0
    logger.info(f"  TF-IDF fit_transform took {vectorize_elapsed:.1f}s")
    
    del all_texts
    gc.collect()
    
    s1_tfidf = all_tfidf[:len(s1_texts)].tocsr()
    s23_tfidf = all_tfidf[len(s1_texts):].tocsr()
    
    logger.info(f"  TF-IDF matrix: {all_tfidf.shape[0]} docs × {all_tfidf.shape[1]} features")
    logger.info(f"  S1 nnz: {s1_tfidf.nnz:,}  S23 nnz: {s23_tfidf.nnz:,}")

    del all_tfidf
    gc.collect()

    def _fmt_duration(seconds: float) -> str:
        if seconds < 0 or seconds != seconds:  # negative or NaN guard
            return "unknown"
        seconds = int(round(seconds))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h}h{m:02d}m{s:02d}s"
        if m:
            return f"{m}m{s:02d}s"
        return f"{s}s"

    def _top_terms_per_row(mat: csr_matrix, k: int) -> List[np.ndarray]:
        """For each row, return the column indices of its top-k highest-weight
        (most distinctive) entries. Much cheaper than using all nonzeros."""
        indptr, indices, data = mat.indptr, mat.indices, mat.data
        out = []
        for i in range(mat.shape[0]):
            start, end = indptr[i], indptr[i + 1]
            row_len = end - start
            if row_len == 0:
                out.append(np.empty(0, dtype=indices.dtype))
                continue
            row_idx = indices[start:end]
            row_data = data[start:end]
            if row_len <= k:
                out.append(row_idx)
            else:
                part = np.argpartition(row_data, -k)[-k:]
                out.append(row_idx[part])
        return out

    n_s23 = s23_tfidf.shape[0]
    effective_top_k = min(top_k, n_s23)

    # ------------------------------------------------------------------
    # 1. Build an inverted index over S23's most distinctive terms.
    # ------------------------------------------------------------------
    logger.info(
        f"  Building inverted index over S2/S3 top-{max_terms_per_doc} terms/doc "
        f"(max posting list size={max_posting_list_size:,})..."
    )
    index_t0 = time.time()

    s23_top_terms = _top_terms_per_row(s23_tfidf, max_terms_per_doc)

    term_postings: Dict[int, List[int]] = defaultdict(list)
    for row_idx, terms in enumerate(s23_top_terms):
        for term in terms:
            term_postings[int(term)].append(row_idx)

    # Drop overly common terms (their posting list is too big to be selective
    # and would dominate query cost without adding precision).
    dropped_terms = 0
    for term in list(term_postings.keys()):
        if len(term_postings[term]) > max_posting_list_size:
            del term_postings[term]
            dropped_terms += 1

    index_elapsed = time.time() - index_t0
    logger.info(
        f"  Inverted index built in {index_elapsed:.1f}s: "
        f"{len(term_postings):,} terms kept, {dropped_terms:,} overly common terms dropped"
    )

    # ------------------------------------------------------------------
    # 2. For each S1 doc, gather a small candidate pool via the inverted
    #    index, then compute exact cosine only within that pool.
    # ------------------------------------------------------------------
    candidates: Dict[str, Set[str]] = {}

    s1_top_terms = _top_terms_per_row(s1_tfidf, max_terms_per_doc)

    logger.info(f"  Querying {len(s1_ids):,} S1 entities against inverted index...")

    search_t0 = time.time()
    query_batch_size = max(1, min(batch_size, len(s1_ids)))
    n_batches = (len(s1_ids) + query_batch_size - 1) // query_batch_size
    total_pool_size = 0
    empty_pools = 0

    for batch_idx, batch_start in enumerate(range(0, len(s1_ids), query_batch_size)):
        batch_end = min(batch_start + query_batch_size, len(s1_ids))
        batch_t0 = time.time()

        for i in range(batch_start, batch_end):
            s1_id = s1_ids[i]

            # Gather this doc's candidate-term postings, sorted rarest-first
            # (smallest posting list = most discriminative term). We consume
            # terms in that order and stop once the pool hits its hard cap,
            # so per-query cost is bounded regardless of how many moderately
            # common terms this doc happens to have.
            term_lists = []
            for term in s1_top_terms[i]:
                postings = term_postings.get(int(term))
                if postings:
                    term_lists.append(postings)
            term_lists.sort(key=len)

            pool_idx_set: Set[int] = set()
            for postings in term_lists:
                pool_idx_set.update(postings)
                if len(pool_idx_set) >= max_candidate_pool_size:
                    break

            if not pool_idx_set:
                candidates[s1_id] = set()
                empty_pools += 1
                continue

            if len(pool_idx_set) > max_candidate_pool_size:
                # Trim down to the cap (union of a couple term postings can
                # overshoot it); order doesn't matter here since we still
                # rank by exact cosine similarity next.
                pool_idx_set = set(list(pool_idx_set)[:max_candidate_pool_size])

            if not pool_idx_set:
                candidates[s1_id] = set()
                empty_pools += 1
                continue

            pool_idx = np.fromiter(pool_idx_set, dtype=np.int64, count=len(pool_idx_set))
            total_pool_size += len(pool_idx)

            # Exact cosine similarity, but only against the small pool
            # (vectors are L2-normalized, so dot product = cosine similarity).
            pool_matrix = s23_tfidf[pool_idx]
            query_vec = s1_tfidf[i]
            sims = pool_matrix.dot(query_vec.T).toarray().ravel()

            k = min(effective_top_k, len(sims))
            if k <= 0:
                candidates[s1_id] = set()
                continue
            if len(sims) <= k:
                top_local = np.argsort(-sims)
            else:
                part = np.argpartition(sims, -k)[-k:]
                top_local = part[np.argsort(-sims[part])]

            top_local = top_local[sims[top_local] > 0]
            top_global = pool_idx[top_local]
            candidates[s1_id] = {s23_ids[idx] for idx in top_global}

        batch_elapsed = time.time() - batch_t0
        rows_done = batch_end
        elapsed = time.time() - search_t0
        rows_per_sec = rows_done / elapsed if elapsed > 0 else 0.0
        remaining_rows = len(s1_ids) - rows_done
        eta_seconds = remaining_rows / rows_per_sec if rows_per_sec > 0 else float("nan")

        if batch_idx < 3 or (batch_idx % 10 == 0) or rows_done == len(s1_ids):
            avg_pool = total_pool_size / max(rows_done - empty_pools, 1)
            logger.info(
                f"  [{rows_done:,}/{len(s1_ids):,}] "
                f"batch {batch_idx + 1}/{n_batches} took {batch_elapsed:.2f}s | "
                f"{rows_per_sec:.1f} rows/s | "
                f"avg candidate pool {avg_pool:.0f} | "
                f"elapsed {_fmt_duration(elapsed)} | "
                f"ETA {_fmt_duration(eta_seconds)}"
            )

    total_search_elapsed = time.time() - search_t0
    logger.info(
        f"  Inverted-index search complete: {len(s1_ids):,} S1 entities "
        f"in {_fmt_duration(total_search_elapsed)} "
        f"({len(s1_ids) / total_search_elapsed if total_search_elapsed > 0 else 0:.1f} rows/s), "
        f"{empty_pools:,} entities had no candidate pool"
    )

    del s1_tfidf, s23_tfidf, term_postings, s23_top_terms, s1_top_terms
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
    tfidf_max_terms_per_doc: int = 15,
    tfidf_max_posting_list_size: int = 2000,
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
        tfidf_max_terms_per_doc: How many top-weighted terms per doc to index/
            query with in the inverted-index prefilter. Lower = faster, may
            reduce recall.
        tfidf_max_posting_list_size: Drop inverted-index terms whose posting
            list exceeds this many docs (too common to be selective). Lower =
            faster, smaller candidate pools, may reduce recall.
    
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
        max_terms_per_doc=tfidf_max_terms_per_doc,
        max_posting_list_size=tfidf_max_posting_list_size,
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