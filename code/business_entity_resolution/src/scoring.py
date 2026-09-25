"""
scoring.py — F_0.5 scoring and threshold tuning.

Implements:
  - Per-entity F_0.5 computation
  - Macro-averaged F_0.5 across all S1 entities
  - Threshold sweep to find optimal decision threshold against F_0.5
"""

import logging
from typing import Dict, List, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def f_beta_score(
    predicted: Set[str],
    truth: Set[str],
    beta: float = 0.5,
) -> float:
    """
    Compute F_β score for a single Source 1 entity.
    
    Per the problem statement:
    - Singletons (truth is empty): score 1.0 if predicted is empty, else 0.0.
    - Non-singletons: standard F_β formula.
    
    Formula:
        F_β = (1 + β²) × Precision × Recall / (β² × Precision + Recall)
    
    Where:
        Precision = |predicted ∩ truth| / |predicted|
        Recall    = |predicted ∩ truth| / |truth|
    
    Args:
        predicted: Set of predicted matching entity IDs.
        truth: Set of true matching entity IDs.
        beta: Beta parameter (0.5 for this challenge).
    
    Returns:
        F_β score for this entity (float in [0, 1]).
    """
    # Singleton handling per problem spec
    if len(truth) == 0:
        # No true matches — score 1.0 if we correctly predict empty, 0.0 otherwise
        return 1.0 if len(predicted) == 0 else 0.0
    
    # Non-singleton
    if len(predicted) == 0:
        # We predicted no matches but there are true matches → recall = 0
        return 0.0
    
    tp = len(predicted & truth)
    
    if tp == 0:
        return 0.0
    
    precision = tp / len(predicted)
    recall = tp / len(truth)
    
    beta_sq = beta ** 2
    f_score = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
    
    return f_score


def macro_f_beta(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
    beta: float = 0.5,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute macro-averaged F_β score across all Source 1 entities.
    
    Args:
        predictions: S1 entity_id → set of predicted matching S2/S3 entity_ids.
        ground_truth: S1 entity_id → set of true matching S2/S3 entity_ids.
        beta: Beta parameter (0.5 for this challenge).
    
    Returns:
        (macro_f_beta, per_entity_scores) where:
          - macro_f_beta: float, the macro-averaged score
          - per_entity_scores: dict of S1_id → individual F_β score
    """
    per_entity_scores = {}
    
    for s1_id in ground_truth:
        pred = predictions.get(s1_id, set())
        truth = ground_truth[s1_id]
        score = f_beta_score(pred, truth, beta)
        per_entity_scores[s1_id] = score
    
    if not per_entity_scores:
        return 0.0, {}
    
    macro_score = sum(per_entity_scores.values()) / len(per_entity_scores)
    
    return macro_score, per_entity_scores


def threshold_sweep(
    pair_ids: List[Tuple[str, str]],
    probabilities: np.ndarray,
    ground_truth: Dict[str, Set[str]],
    thresholds: np.ndarray = None,
    beta: float = 0.5,
) -> Tuple[float, float, List[Tuple[float, float]]]:
    """
    Sweep over decision thresholds to find the optimal one for F_β.
    
    Args:
        pair_ids: List of (s1_id, s23_id) tuples.
        probabilities: Array of predicted match probabilities, same length as pair_ids.
        ground_truth: S1 entity_id → set of true matching S2/S3 entity_ids.
        thresholds: Array of threshold values to try. If None, uses np.arange(0.1, 0.96, 0.02).
        beta: Beta parameter (0.5).
    
    Returns:
        (best_threshold, best_f_beta, sweep_results) where:
          - best_threshold: float
          - best_f_beta: float
          - sweep_results: list of (threshold, f_beta) tuples
    """
    if thresholds is None:
        thresholds = np.arange(0.10, 0.96, 0.02)
    
    # Get all S1 entities from ground truth
    all_s1_ids = set(ground_truth.keys())
    
    sweep_results = []
    best_threshold = 0.5
    best_f_beta = 0.0
    
    logger.info("Threshold sweep for F_{:.1f}:".format(beta))
    logger.info(f"  {'Threshold':>10}  {'F_beta':>8}  {'Matched':>8}  {'Singletons':>10}")
    logger.info(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*10}")
    
    for threshold in thresholds:
        # Build predictions at this threshold
        predictions: Dict[str, Set[str]] = {s1_id: set() for s1_id in all_s1_ids}
        
        for (s1_id, s23_id), prob in zip(pair_ids, probabilities):
            if prob >= threshold and s1_id in predictions:
                predictions[s1_id].add(s23_id)
        
        # Score
        score, _ = macro_f_beta(predictions, ground_truth, beta)
        
        # Count stats
        n_matched = sum(1 for v in predictions.values() if len(v) > 0)
        n_singletons = sum(1 for v in predictions.values() if len(v) == 0)
        
        sweep_results.append((threshold, score))
        
        if score > best_f_beta:
            best_f_beta = score
            best_threshold = threshold
        
        logger.info(f"  {threshold:>10.3f}  {score:>8.4f}  {n_matched:>8}  {n_singletons:>10}")
    
    logger.info(f"\n  *** Best threshold: {best_threshold:.3f} -> F_{beta:.1f} = {best_f_beta:.4f} ***")
    
    return best_threshold, best_f_beta, sweep_results


def detailed_evaluation(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
    beta: float = 0.5,
) -> Dict[str, float]:
    """
    Detailed evaluation with breakdown by category.
    
    Returns dict with:
      - macro_f_beta
      - singleton_accuracy (% of true singletons correctly predicted empty)
      - non_singleton_f_beta (F_beta only on entities with true matches)
      - precision_macro (macro-averaged precision)
      - recall_macro (macro-averaged recall)
    """
    macro_score, per_entity = macro_f_beta(predictions, ground_truth, beta)
    
    # Breakdown
    singleton_correct = 0
    singleton_total = 0
    non_singleton_scores = []
    precisions = []
    recalls = []
    
    for s1_id, truth in ground_truth.items():
        pred = predictions.get(s1_id, set())
        
        if len(truth) == 0:
            singleton_total += 1
            if len(pred) == 0:
                singleton_correct += 1
        else:
            non_singleton_scores.append(per_entity.get(s1_id, 0.0))
            tp = len(pred & truth)
            if len(pred) > 0:
                precisions.append(tp / len(pred))
            else:
                precisions.append(0.0 if len(truth) > 0 else 1.0)
            recalls.append(tp / len(truth))
    
    results = {
        "macro_f_beta": macro_score,
        "singleton_accuracy": singleton_correct / max(singleton_total, 1),
        "singleton_correct": singleton_correct,
        "singleton_total": singleton_total,
        "non_singleton_f_beta": np.mean(non_singleton_scores) if non_singleton_scores else 0.0,
        "non_singleton_count": len(non_singleton_scores),
        "precision_macro": np.mean(precisions) if precisions else 0.0,
        "recall_macro": np.mean(recalls) if recalls else 0.0,
        "total_entities": len(ground_truth),
        "entities_with_predictions": sum(1 for v in predictions.values() if len(v) > 0),
    }
    
    return results
