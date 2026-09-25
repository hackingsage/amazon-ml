#!/usr/bin/env python3
"""
run_all.py — One-click execution for all prerequisites and pipeline stages.

Performs:
  1. Dependency Verification: Checks and auto-installs requirements.txt if needed.
  2. Test Suite Execution: Runs unit tests (normalization, scoring, features).
  3. Dataset Integrity Check: Verifies train/test TSVs (auto-generates synthetic data if missing).
  4. End-to-End Pipeline Execution: Runs blocking, training, threshold tuning, and inference.
  5. Submission Validation: Runs official submission validator (utils/validate_submission.py).

Usage:
  python run_all.py                  # Run all prerequisites + train + inference + validation
  python run_all.py --mode train     # Run prerequisites + train only
  python run_all.py --mode test      # Run prerequisites + test inference only
  python run_all.py --synthetic      # Force regenerate synthetic benchmark data
  python run_all.py --skip-install   # Skip checking/installing pip dependencies
"""

import argparse
import importlib
import os
import subprocess
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_banner(text: str):
    line = "=" * 65
    print(f"\n{line}")
    print(f" {text}")
    print(f"{line}")


def check_and_install_dependencies():
    """Verify core packages and auto-install from requirements.txt if missing."""
    print_banner("[1/5] Checking Dependencies")
    
    required_pkgs = ["pandas", "numpy", "scipy", "sklearn", "lightgbm"]
    missing = []
    
    for pkg in required_pkgs:
        try:
            importlib.import_module(pkg)
            print(f"  [OK] Found {pkg}")
        except ImportError:
            missing.append(pkg)
            print(f"  [MISSING] {pkg}")
            
    if missing:
        req_file = PROJECT_ROOT / "requirements.txt"
        print(f"\nMissing packages detected ({', '.join(missing)}).")
        print(f"Auto-installing dependencies from {req_file}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_file)])
            print("  [OK] Dependencies installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] Failed to install dependencies: {e}")
            sys.exit(1)
    else:
        print("  [OK] All core dependencies satisfied.")


def run_unit_tests():
    """Run the unit test suite covering normalization, scoring, and features."""
    print_banner("[2/5] Running Built-in Test Suite")
    
    test_script = PROJECT_ROOT / "code" / "business_entity_resolution" / "src" / "test_components.py"
    cmd = [sys.executable, str(test_script)]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout.strip())
        print("  [PASS] All unit tests passed (normalization, F_0.5 formula, features).")
    else:
        print(result.stdout)
        print(result.stderr)
        print("  [FAIL] Unit tests failed. Aborting pipeline.")
        sys.exit(1)


def check_dataset_and_fallback(force_synthetic: bool = False):
    """Verify that dataset files exist; generate synthetic data if missing."""
    print_banner("[3/5] Checking Dataset Availability")
    
    required_train_files = [
        PROJECT_ROOT / "dataset" / "train" / "train_source1.tsv",
        PROJECT_ROOT / "dataset" / "train" / "train_source2.tsv",
        PROJECT_ROOT / "dataset" / "train" / "train_source3.tsv",
        PROJECT_ROOT / "dataset" / "train" / "train_ground_truth.tsv",
    ]
    required_test_files = [
        PROJECT_ROOT / "dataset" / "test" / "test_source1.tsv",
        PROJECT_ROOT / "dataset" / "test" / "test_source2.tsv",
        PROJECT_ROOT / "dataset" / "test" / "test_source3.tsv",
    ]
    
    all_files = required_train_files + required_test_files
    missing_files = [f for f in all_files if not f.exists() or f.stat().st_size == 0]
    
    if force_synthetic or missing_files:
        if force_synthetic:
            print("  Flag --synthetic passed: Regenerating synthetic dataset...")
        else:
            print("  Official dataset files not found in dataset/train/ or dataset/test/.")
            print("  Auto-generating synthetic benchmark data (US, India, France) for testing...")
        
        synth_script = PROJECT_ROOT / "create_synthetic_data.py"
        subprocess.check_call([sys.executable, str(synth_script)])
        print("  [OK] Benchmark dataset ready.")
    else:
        print("  [OK] Valid competition dataset files detected:")
        print(f"       Train files: {len(required_train_files)} files in dataset/train/")
        print(f"       Test files:  {len(required_test_files)} files in dataset/test/")


def run_pipeline(mode: str = "both", val_fraction: float = 0.2):
    """Execute the pipeline orchestrator."""
    print_banner(f"[4/5] Executing Pipeline (Mode: {mode})")
    
    pipeline_cmd = [
        sys.executable,
        "-m", "code.business_entity_resolution.src.pipeline",
        "--mode", mode,
        "--val-fraction", str(val_fraction),
    ]
    
    start_time = time.time()
    result = subprocess.run(pipeline_cmd)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"  [ERROR] Pipeline failed with exit code {result.returncode}.")
        sys.exit(result.returncode)
    
    print(f"  [OK] Pipeline completed in {elapsed:.1f}s.")


def validate_outputs():
    """Run the official submission validator."""
    print_banner("[5/5] Validating Submission Outputs")
    
    validator_script = PROJECT_ROOT / "utils" / "validate_submission.py"
    matching_file = PROJECT_ROOT / "output" / "matching_results.tsv"
    candidate_file = PROJECT_ROOT / "output" / "candidate_pairs.tsv"
    test_dir = PROJECT_ROOT / "dataset" / "test"
    
    val_cmd = [
        sys.executable, str(validator_script),
        "--matching", str(matching_file),
        "--candidate", str(candidate_file),
        "--test-dir", str(test_dir),
    ]
    
    result = subprocess.run(val_cmd)
    if result.returncode == 0:
        print("\n" + "=" * 65)
        print(" ALL PREREQUISITES & PIPELINE STAGES PASSED SUCCESSFULLY!")
        print("=" * 65)
        print("Generated submission files:")
        print(f"  - Matching results:  {matching_file.relative_to(PROJECT_ROOT)}")
        print(f"  - Candidate pairs:   {candidate_file.relative_to(PROJECT_ROOT)}")
        print("\nNext steps:")
        print("  1. Upload output/matching_results.tsv to the challenge portal.")
        print("  2. To package the final submission zip archive, run:")
        print("     powershell: Compress-Archive output, code, Documentation_template.md -DestinationPath submission.zip")
    else:
        print("  [ERROR] Output validation failed. Check log messages above.")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="One-click execution for all prerequisites and pipeline stages.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["train", "test", "both"],
        default="both",
        help="Pipeline execution mode: 'both' (retrain + test inference, default), 'train', or 'test'",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Validation split fraction for threshold tuning (default: 0.2)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Force regeneration of synthetic sample dataset",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip checking/installing pip requirements",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running unit tests",
    )
    
    args = parser.parse_args()
    total_start = time.time()
    
    print("\n" + "=" * 65)
    print(" Business Entity Resolution Pipeline - Full Prerequisite & Runner")
    print("=" * 65)
    
    # 1. Dependencies
    if not args.skip_install:
        check_and_install_dependencies()
    
    # 2. Unit tests
    if not args.skip_tests:
        run_unit_tests()
    
    # 3. Dataset check / fallback
    check_dataset_and_fallback(force_synthetic=args.synthetic)
    
    # 4. Pipeline execution
    run_pipeline(mode=args.mode, val_fraction=args.val_fraction)
    
    # 5. Validation (if mode produced test outputs)
    if args.mode in ("test", "both"):
        validate_outputs()
    
    total_elapsed = time.time() - total_start
    print(f"\nTotal elapsed execution time: {total_elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
