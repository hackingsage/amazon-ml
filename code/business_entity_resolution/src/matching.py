"""
matching.py — Matching model: LightGBM gradient-boosted trees.

Trains a binary classifier on pairwise similarity features to predict
match / no-match. Supports:
  - Training with label generation from ground truth
  - Threshold tuning against F_0.5
  - Inference on test candidate pairs
  - (Placeholder) LLM tiebreaker for hybrid mode

The GBT model biases toward precision (fewer false merges) since F_0.5
weighs precision 2× over recall.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from .scoring import threshold_sweep, macro_f_beta, detailed_evaluation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------

def generate_labels(
    pair_ids: List[Tuple[str, str]],
    ground_truth: Dict[str, Set[str]],
) -> np.ndarray:
    """
    Generate binary labels for candidate pairs based on ground truth.
    
    Args:
        pair_ids: List of (s1_id, s23_id) tuples.
        ground_truth: S1 entity_id → set of true matching S2/S3 entity_ids.
    
    Returns:
        Binary label array (1 = match, 0 = no match).
    """
    labels = np.zeros(len(pair_ids), dtype=np.float32)
    for i, (s1_id, s23_id) in enumerate(pair_ids):
        truth = ground_truth.get(s1_id, set())
        if s23_id in truth:
            labels[i] = 1.0
    
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    logger.info(f"Labels: {n_pos} positives ({100*n_pos/max(len(labels),1):.1f}%), "
                f"{n_neg} negatives ({100*n_neg/max(len(labels),1):.1f}%)")
    
    return labels


# ---------------------------------------------------------------------------
# LightGBM training
# ---------------------------------------------------------------------------

def train_lightgbm(
    feature_df: pd.DataFrame,
    labels: np.ndarray,
    pair_ids: List[Tuple[str, str]],
    ground_truth: Dict[str, Set[str]],
    val_feature_df: Optional[pd.DataFrame] = None,
    val_labels: Optional[np.ndarray] = None,
    val_pair_ids: Optional[List[Tuple[str, str]]] = None,
    val_ground_truth: Optional[Dict[str, Set[str]]] = None,
    model_save_path: Optional[str] = None,
) -> Tuple[lgb.Booster, float]:
    """
    Train LightGBM binary classifier on pairwise features.
    
    Args:
        feature_df: Training feature matrix.
        labels: Training binary labels.
        pair_ids: Training pair IDs (for logging).
        ground_truth: Training ground truth (for logging).
        val_feature_df: Validation feature matrix (optional).
        val_labels: Validation labels (optional).
        val_pair_ids: Validation pair IDs (for threshold tuning).
        val_ground_truth: Validation ground truth (for threshold tuning).
        model_save_path: Path to save trained model (optional).
    
    Returns:
        (model, best_threshold) tuple.
    """
    logger.info("=" * 60)
    logger.info("MATCHING STAGE: Training LightGBM")
    logger.info("=" * 60)
    
    # Compute scale_pos_weight for class imbalance
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)
    
    logger.info(f"Class balance: {n_pos} pos / {n_neg} neg → scale_pos_weight={scale_pos_weight:.2f}")
    
    # LightGBM parameters — tuned for precision on entity resolution
    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "max_depth": 8,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": scale_pos_weight,
        "verbose": -1,
        "random_state": 42,
        "n_jobs": -1,
    }
    
    logger.info(f"LightGBM params: {params}")
    
    # Create datasets
    train_data = lgb.Dataset(feature_df, label=labels)
    
    valid_sets = [train_data]
    valid_names = ["train"]
    
    if val_feature_df is not None and val_labels is not None:
        val_data = lgb.Dataset(val_feature_df, label=val_labels, reference=train_data)
        valid_sets.append(val_data)
        valid_names.append("valid")
    
    # Train
    callbacks = [lgb.log_evaluation(period=50)]
    if val_feature_df is not None and val_labels is not None:
        callbacks.append(lgb.early_stopping(stopping_rounds=50, verbose=True))
    
    n_rounds = params.pop("n_estimators", 500)
    model = lgb.train(
        params,
        train_data,
        num_boost_round=n_rounds,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    
    best_iter = model.best_iteration if (model.best_iteration and model.best_iteration > 0) else model.num_trees()
    logger.info(f"Training complete. Best iteration: {best_iter}")
    
    # Feature importance
    importance = model.feature_importance(importance_type="gain")
    feature_names = feature_df.columns.tolist()
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False)
    
    logger.info("\nTop 15 features by gain:")
    for _, row in imp_df.head(15).iterrows():
        logger.info(f"  {row['feature']:<35} {row['importance']:.1f}")
    
    # --- Threshold tuning on validation set ---
    best_threshold = 0.5  # default
    
    if val_feature_df is not None and val_pair_ids is not None and val_ground_truth is not None:
        logger.info("\n" + "=" * 60)
        logger.info("THRESHOLD TUNING on validation split")
        logger.info("=" * 60)
        
        num_iter = model.best_iteration if (model.best_iteration and model.best_iteration > 0) else None
        val_probs = model.predict(val_feature_df, num_iteration=num_iter)
        
        best_threshold, best_f05, sweep = threshold_sweep(
            val_pair_ids, val_probs, val_ground_truth, beta=0.5,
        )
        
        # Detailed evaluation at best threshold
        val_predictions: Dict[str, Set[str]] = {s1_id: set() for s1_id in val_ground_truth}
        for (s1_id, s23_id), prob in zip(val_pair_ids, val_probs):
            if prob >= best_threshold and s1_id in val_predictions:
                val_predictions[s1_id].add(s23_id)
        
        details = detailed_evaluation(val_predictions, val_ground_truth, beta=0.5)
        
        logger.info("\nDetailed validation results at threshold {:.3f}:".format(best_threshold))
        for k, v in details.items():
            if isinstance(v, float):
                logger.info(f"  {k:<30} {v:.4f}")
            else:
                logger.info(f"  {k:<30} {v}")
    
    # Save model
    if model_save_path:
        Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
        model.save_model(model_save_path)
        logger.info(f"\nModel saved to {model_save_path}")
    
    return model, best_threshold


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_matches(
    model: lgb.Booster,
    feature_df: pd.DataFrame,
    pair_ids: List[Tuple[str, str]],
    threshold: float,
    all_s1_ids: Set[str],
) -> Dict[str, Set[str]]:
    """
    Run inference: predict match probabilities and apply threshold.
    
    Args:
        model: Trained LightGBM model.
        feature_df: Feature matrix for candidate pairs.
        pair_ids: List of (s1_id, s23_id) tuples.
        threshold: Decision threshold.
        all_s1_ids: Set of all S1 entity IDs (to ensure every S1 appears in output).
    
    Returns:
        Dict mapping S1 entity_id → set of matched S2/S3 entity_ids.
    """
    logger.info(f"Running inference on {len(pair_ids)} candidate pairs (threshold={threshold:.3f})...")
    
    # Initialize all S1 entities with empty predictions
    predictions: Dict[str, Set[str]] = {s1_id: set() for s1_id in all_s1_ids}
    
    if len(feature_df) == 0:
        logger.warning("Empty feature matrix — all predictions will be empty (singletons).")
        return predictions
    
    # Predict probabilities
    num_iter = model.best_iteration if (model.best_iteration and model.best_iteration > 0) else None
    probs = model.predict(feature_df, num_iteration=num_iter)
    
    # Apply threshold
    n_matches = 0
    for (s1_id, s23_id), prob in zip(pair_ids, probs):
        if prob >= threshold:
            if s1_id in predictions:
                predictions[s1_id].add(s23_id)
                n_matches += 1
    
    n_matched_entities = sum(1 for v in predictions.values() if len(v) > 0)
    n_singletons = sum(1 for v in predictions.values() if len(v) == 0)
    
    logger.info(f"  Total matches: {n_matches}")
    logger.info(f"  S1 entities with matches: {n_matched_entities}")
    logger.info(f"  S1 singletons (no match): {n_singletons}")
    
    return predictions


# ---------------------------------------------------------------------------
# LLM tiebreaker (placeholder for hybrid mode)
# ---------------------------------------------------------------------------

def llm_tiebreaker(
    model: lgb.Booster,
    feature_df: pd.DataFrame,
    pair_ids: List[Tuple[str, str]],
    probabilities: np.ndarray,
    threshold: float,
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    ambiguity_margin: float = 0.10,
) -> Dict[str, Set[str]]:
    """
    Hybrid mode: use LLM to break ties on ambiguous pairs.
    
    This is a placeholder — implement when you have an LLM model available.
    Currently returns the GBT-only predictions.
    
    Args:
        model: Trained GBT model.
        feature_df: Feature matrix.
        pair_ids: Pair IDs.
        probabilities: GBT probabilities.
        threshold: GBT threshold.
        s1_df: Source 1 dataframe (for prompt construction).
        s23_df: Source 2/3 dataframe (for prompt construction).
        ambiguity_margin: Margin around threshold to consider "ambiguous".
    
    Returns:
        Final predictions dict.
    """
    logger.warning("LLM tiebreaker not implemented yet — using GBT-only predictions.")
    logger.info(f"  Ambiguous pairs (within ±{ambiguity_margin} of threshold {threshold:.3f}): "
                f"{np.sum(np.abs(probabilities - threshold) < ambiguity_margin)}")
    
    # TODO: Implement LLM tiebreaker
    # 1. Filter pairs where |prob - threshold| < ambiguity_margin
    # 2. For each ambiguous pair, construct prompt with S1 and S23 records
    # 3. Run LLM inference (local, e.g. via llama-cpp-python or transformers)
    # 4. Parse LLM output for match/no-match decision
    # 5. Override GBT decision with LLM decision for ambiguous pairs
    
    return {}  # caller should fall back to GBT-only
