"""
pipeline.py — Main pipeline runner for Business Entity Resolution.

Usage:
    python -m code.business_entity_resolution.src.pipeline --mode train
    python -m code.business_entity_resolution.src.pipeline --mode test
    python -m code.business_entity_resolution.src.pipeline --mode both

Modes:
  - train: Train + validate on a held-out split from training data.
           Reports blocking recall ceiling and F_0.5 score.
  - test:  Run inference on test data using a pre-trained model.
           Produces output/matching_results.tsv and output/candidate_pairs.tsv.
  - both:  Train on full training data, then run inference on test data.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Ensure the parent directories are on the Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from code.business_entity_resolution.src.blocking import (
    union_blocking,
    evaluate_blocking,
)
from code.business_entity_resolution.src.features import compute_features_for_pairs
from code.business_entity_resolution.src.matching import (
    generate_labels,
    train_lightgbm,
    predict_matches,
)
from code.business_entity_resolution.src.inference import (
    write_matching_results,
    write_candidate_pairs,
    validate_submission,
)
from code.business_entity_resolution.src.scoring import (
    macro_f_beta,
    detailed_evaluation,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_file: str = None):
    """Configure logging to console and optionally to file."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w", encoding="utf-8"))
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_source_data(path: str) -> pd.DataFrame:
    """Load a source TSV file with proper tab separator."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    # Verify we got the expected columns (not a single column from wrong sep)
    if len(df.columns) == 1:
        raise ValueError(
            f"Only 1 column found in {path} — did you use sep='\\t'? "
            f"Column: {df.columns[0]}"
        )
    logging.info(f"  Loaded {path}: {len(df)} rows, columns={list(df.columns)}")
    return df


def load_ground_truth(path: str) -> Dict[str, Set[str]]:
    """
    Load ground truth TSV: source1_entity_id → set of matched entity_ids.
    Empty matched_entity_ids → empty set (singleton).
    """
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    gt: Dict[str, Set[str]] = {}
    for _, row in df.iterrows():
        s1_id = row["source1_entity_id"]
        matched_str = str(row.get("matched_entity_ids", ""))
        if matched_str:
            matched_ids = set(x.strip() for x in matched_str.split(",") if x.strip())
        else:
            matched_ids = set()
        gt[s1_id] = matched_ids
    
    n_singletons = sum(1 for v in gt.values() if len(v) == 0)
    n_matched = sum(1 for v in gt.values() if len(v) > 0)
    total_matches = sum(len(v) for v in gt.values())
    logging.info(f"  Ground truth: {len(gt)} entities, {n_matched} with matches "
                 f"({total_matches} total match IDs), {n_singletons} singletons")
    return gt


# ---------------------------------------------------------------------------
# Train/validation split
# ---------------------------------------------------------------------------

def split_train_val(
    s1_df: pd.DataFrame,
    ground_truth: Dict[str, Set[str]],
    val_fraction: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Split S1 entities into train and validation sets.
    
    We split at the S1 entity level (not pair level) to ensure
    validation evaluates complete entities.
    """
    s1_ids = s1_df["entity_id"].tolist()
    np.random.seed(random_state)
    np.random.shuffle(s1_ids)
    
    n_val = max(1, int(len(s1_ids) * val_fraction))
    val_ids = set(s1_ids[:n_val])
    train_ids = set(s1_ids[n_val:])
    
    train_s1 = s1_df[s1_df["entity_id"].isin(train_ids)].reset_index(drop=True)
    val_s1 = s1_df[s1_df["entity_id"].isin(val_ids)].reset_index(drop=True)
    
    train_gt = {k: v for k, v in ground_truth.items() if k in train_ids}
    val_gt = {k: v for k, v in ground_truth.items() if k in val_ids}
    
    logging.info(f"  Train/val split: {len(train_s1)} train, {len(val_s1)} val "
                 f"(val_fraction={val_fraction})")
    
    return train_s1, val_s1, train_gt, val_gt


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def run_training_pipeline(
    data_dir: str = "dataset/train",
    output_dir: str = "output",
    model_dir: str = "code/business_entity_resolution/models",
    val_fraction: float = 0.2,
    top_k: int = 30,
    max_block_size: int = 500,
):
    """
    Full training pipeline:
    1. Load data
    2. Split train/val
    3. Blocking
    4. Feature computation
    5. Model training + threshold tuning
    6. Validation evaluation
    """
    logger = logging.getLogger(__name__)
    start_time = time.time()
    
    # --- 1. Load data ---
    logger.info("=" * 60)
    logger.info("STEP 1: Loading training data")
    logger.info("=" * 60)
    
    s1_path = os.path.join(data_dir, "train_source1.tsv")
    s2_path = os.path.join(data_dir, "train_source2.tsv")
    s3_path = os.path.join(data_dir, "train_source3.tsv")
    gt_path = os.path.join(data_dir, "train_ground_truth.tsv")
    
    s1_df = load_source_data(s1_path)
    s2_df = load_source_data(s2_path)
    s3_df = load_source_data(s3_path)
    ground_truth = load_ground_truth(gt_path)
    
    # Combine S2 + S3
    s23_df = pd.concat([s2_df, s3_df], ignore_index=True)
    logger.info(f"  Combined S2+S3: {len(s23_df)} records")
    
    # --- 2. Train/val split ---
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Train/validation split")
    logger.info("=" * 60)
    
    train_s1, val_s1, train_gt, val_gt = split_train_val(
        s1_df, ground_truth, val_fraction=val_fraction,
    )
    
    # --- 3. Blocking ---
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Blocking (candidate generation)")
    logger.info("=" * 60)
    
    # Run blocking on FULL S23 (both train and val S1 entities need candidates from all S23)
    train_candidates = union_blocking(train_s1, s23_df, top_k=top_k, max_block_size=max_block_size)
    val_candidates = union_blocking(val_s1, s23_df, top_k=top_k, max_block_size=max_block_size)
    
    # Evaluate blocking on validation split
    total_s23 = len(s23_df)
    blocking_metrics = evaluate_blocking(val_candidates, val_gt, total_s23)
    
    logger.info("\nBlocking evaluation (validation split):")
    for k, v in blocking_metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k:<35} {v:.4f}")
        else:
            logger.info(f"  {k:<35} {v}")
    
    # --- 4. Feature computation ---
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Feature computation")
    logger.info("=" * 60)
    
    train_features, train_pair_ids = compute_features_for_pairs(
        train_s1, s23_df, train_candidates,
    )
    val_features, val_pair_ids = compute_features_for_pairs(
        val_s1, s23_df, val_candidates,
    )
    
    # --- 5. Training + threshold tuning ---
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Model training + threshold tuning")
    logger.info("=" * 60)
    
    train_labels = generate_labels(train_pair_ids, train_gt)
    val_labels = generate_labels(val_pair_ids, val_gt)
    
    model_path = os.path.join(model_dir, "lightgbm_model.txt")
    
    model, best_threshold = train_lightgbm(
        train_features, train_labels, train_pair_ids, train_gt,
        val_feature_df=val_features,
        val_labels=val_labels,
        val_pair_ids=val_pair_ids,
        val_ground_truth=val_gt,
        model_save_path=model_path,
    )
    
    # --- 6. Final validation evaluation ---
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: Final validation evaluation")
    logger.info("=" * 60)
    
    val_predictions = predict_matches(
        model, val_features, val_pair_ids, best_threshold,
        all_s1_ids=set(val_gt.keys()),
    )
    
    details = detailed_evaluation(val_predictions, val_gt, beta=0.5)
    logger.info("\nFinal validation results:")
    for k, v in details.items():
        if isinstance(v, float):
            logger.info(f"  {k:<30} {v:.4f}")
        else:
            logger.info(f"  {k:<30} {v}")
    
    # Save threshold
    threshold_path = os.path.join(model_dir, "threshold.txt")
    Path(threshold_path).parent.mkdir(parents=True, exist_ok=True)
    with open(threshold_path, "w") as f:
        f.write(str(best_threshold))
    logger.info(f"\nThreshold saved: {best_threshold:.4f} -> {threshold_path}")
    
    elapsed = time.time() - start_time
    logger.info(f"\nTraining pipeline complete in {elapsed:.1f}s")
    
    return model, best_threshold


def run_test_pipeline(
    data_dir: str = "dataset/test",
    train_data_dir: str = "dataset/train",
    output_dir: str = "output",
    model_dir: str = "code/business_entity_resolution/models",
    top_k: int = 30,
    max_block_size: int = 500,
    retrain: bool = True,
    train_val_fraction: float = 0.0,  # 0 = use all training data
):
    """
    Test inference pipeline:
    1. Optionally retrain on full training data
    2. Load test data
    3. Blocking on test data
    4. Feature computation
    5. Model inference
    6. Write outputs
    7. Validate outputs
    """
    logger = logging.getLogger(__name__)
    start_time = time.time()
    
    import lightgbm as lgb
    
    # --- Retrain or load model ---
    if retrain:
        logger.info("Retraining on FULL training data (no validation holdout)...")
        
        # Load full training data
        s1_train = load_source_data(os.path.join(train_data_dir, "train_source1.tsv"))
        s2_train = load_source_data(os.path.join(train_data_dir, "train_source2.tsv"))
        s3_train = load_source_data(os.path.join(train_data_dir, "train_source3.tsv"))
        gt = load_ground_truth(os.path.join(train_data_dir, "train_ground_truth.tsv"))
        s23_train = pd.concat([s2_train, s3_train], ignore_index=True)
        
        # If we want a small val split for threshold tuning even in test mode
        if train_val_fraction > 0:
            train_s1, val_s1, train_gt, val_gt = split_train_val(
                s1_train, gt, val_fraction=train_val_fraction,
            )
        else:
            train_s1 = s1_train
            val_s1 = None
            train_gt = gt
            val_gt = None
        
        # Blocking on training data
        train_candidates = union_blocking(train_s1, s23_train, top_k=top_k, max_block_size=max_block_size)
        train_features, train_pair_ids = compute_features_for_pairs(train_s1, s23_train, train_candidates)
        train_labels = generate_labels(train_pair_ids, train_gt)
        
        val_features, val_pair_ids, val_labels_arr = None, None, None
        if val_s1 is not None and val_gt is not None:
            val_candidates = union_blocking(val_s1, s23_train, top_k=top_k, max_block_size=max_block_size)
            val_features, val_pair_ids = compute_features_for_pairs(val_s1, s23_train, val_candidates)
            val_labels_arr = generate_labels(val_pair_ids, val_gt)
        
        model_path = os.path.join(model_dir, "lightgbm_model.txt")
        model, best_threshold = train_lightgbm(
            train_features, train_labels, train_pair_ids, train_gt,
            val_feature_df=val_features,
            val_labels=val_labels_arr,
            val_pair_ids=val_pair_ids,
            val_ground_truth=val_gt,
            model_save_path=model_path,
        )
    else:
        # Load pre-trained model
        model_path = os.path.join(model_dir, "lightgbm_model.txt")
        threshold_path = os.path.join(model_dir, "threshold.txt")
        
        logger.info(f"Loading pre-trained model from {model_path}")
        model = lgb.Booster(model_file=model_path)
        
        with open(threshold_path, "r") as f:
            best_threshold = float(f.read().strip())
        logger.info(f"Loaded threshold: {best_threshold:.4f}")
    
    # --- Load test data ---
    logger.info("\n" + "=" * 60)
    logger.info("Loading test data")
    logger.info("=" * 60)
    
    test_s1 = load_source_data(os.path.join(data_dir, "test_source1.tsv"))
    test_s2 = load_source_data(os.path.join(data_dir, "test_source2.tsv"))
    test_s3 = load_source_data(os.path.join(data_dir, "test_source3.tsv"))
    test_s23 = pd.concat([test_s2, test_s3], ignore_index=True)
    
    test_s1_ids = set(test_s1["entity_id"].tolist())
    test_s23_ids = set(test_s23["entity_id"].tolist())
    
    logger.info(f"Test S1 entities: {len(test_s1_ids)}")
    logger.info(f"Test S2+S3 entities: {len(test_s23_ids)}")
    
    # --- Blocking on test data ---
    logger.info("\n" + "=" * 60)
    logger.info("Blocking on test data")
    logger.info("=" * 60)
    
    test_candidates = union_blocking(test_s1, test_s23, top_k=top_k, max_block_size=max_block_size)
    
    # --- Feature computation ---
    logger.info("\n" + "=" * 60)
    logger.info("Feature computation on test data")
    logger.info("=" * 60)
    
    test_features, test_pair_ids = compute_features_for_pairs(test_s1, test_s23, test_candidates)
    
    # --- Inference ---
    logger.info("\n" + "=" * 60)
    logger.info("Model inference on test data")
    logger.info("=" * 60)
    
    test_predictions = predict_matches(
        model, test_features, test_pair_ids, best_threshold, test_s1_ids,
    )
    
    # --- Write outputs ---
    logger.info("\n" + "=" * 60)
    logger.info("Writing output files")
    logger.info("=" * 60)
    
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
    
    write_matching_results(test_predictions, matching_path)
    write_candidate_pairs(test_candidates, candidate_path)
    
    # --- Validate ---
    is_valid = validate_submission(matching_path, candidate_path, test_s1_ids, test_s23_ids)
    
    elapsed = time.time() - start_time
    logger.info(f"\nTest pipeline complete in {elapsed:.1f}s")
    
    if is_valid:
        logger.info("[SUCCESS] Submission is VALID and ready to upload!")
    else:
        logger.error("[ERROR] Submission has issues - fix them before uploading.")
    
    return is_valid


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Business Entity Resolution Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["train", "test", "both"],
        default="train",
        help="Pipeline mode: 'train' (train+validate), 'test' (inference), 'both' (train then test)",
    )
    parser.add_argument("--train-data", default="dataset/train", help="Path to training data directory")
    parser.add_argument("--test-data", default="dataset/test", help="Path to test data directory")
    parser.add_argument("--output-dir", default="output", help="Path to output directory")
    parser.add_argument("--model-dir", default="code/business_entity_resolution/models", help="Path to model directory")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Validation split fraction (train mode)")
    parser.add_argument("--top-k", type=int, default=30, help="Top-K for TF-IDF blocking")
    parser.add_argument("--max-block-size", type=int, default=500, help="Max block size for token blocking")
    parser.add_argument("--log-file", default=None, help="Log file path (optional)")
    
    args = parser.parse_args()
    
    setup_logging(args.log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("Business Entity Resolution Pipeline")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Train data: {args.train_data}")
    logger.info(f"Test data: {args.test_data}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Model dir: {args.model_dir}")
    
    if args.mode in ("train", "both"):
        run_training_pipeline(
            data_dir=args.train_data,
            output_dir=args.output_dir,
            model_dir=args.model_dir,
            val_fraction=args.val_fraction,
            top_k=args.top_k,
            max_block_size=args.max_block_size,
        )
    
    if args.mode in ("test", "both"):
        run_test_pipeline(
            data_dir=args.test_data,
            train_data_dir=args.train_data,
            output_dir=args.output_dir,
            model_dir=args.model_dir,
            top_k=args.top_k,
            max_block_size=args.max_block_size,
            retrain=(args.mode == "both"),
            train_val_fraction=0.1 if args.mode == "both" else 0.0,
        )


if __name__ == "__main__":
    main()
