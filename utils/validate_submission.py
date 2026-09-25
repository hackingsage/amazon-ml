#!/usr/bin/env python3
"""
utils/validate_submission.py - Standalone submission validator (stdlib only).

Verifies matching_results.tsv and candidate_pairs.tsv against official rules:
1. Both files exist and are readable (.tsv, sep=\\t).
2. Proper headers:
     matching: 'source1_entity_id', 'matched_entity_ids'
     candidate: 'source1_entity_id', 'candidate_entity_ids'
3. Exactly one row per test Source 1 entity; no missing or duplicate S1 rows.
4. matched_entity_ids and candidate_entity_ids are comma-separated IDs without quotes.
5. All IDs in matched/candidate lists belong to S2 or S3 from the test set (no S1 self-matches, no unknown IDs).
6. No duplicate entity IDs within any single list.
7. Candidate completeness: matched_entity_ids ⊆ candidate_entity_ids for every entity.
8. Empty string correctly used for singletons (entities with zero matches/candidates).

Prints PASS (exit 0) or numbered list of issues (exit 1).
"""

import argparse
import csv
import os
import sys
from pathlib import Path


def load_test_entity_ids(test_dir: str):
    """Load valid S1, S2, S3 IDs from test set files using only stdlib csv."""
    s1_file = os.path.join(test_dir, "test_source1.tsv")
    s2_file = os.path.join(test_dir, "test_source2.tsv")
    s3_file = os.path.join(test_dir, "test_source3.tsv")

    for fpath in [s1_file, s2_file, s3_file]:
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing required test file: {fpath}")

    def read_ids(fpath):
        ids = set()
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            if not header or "entity_id" not in header:
                raise ValueError(f"No 'entity_id' column in {fpath}")
            id_idx = header.index("entity_id")
            for row in reader:
                if row and len(row) > id_idx:
                    ids.add(row[id_idx].strip())
        return ids

    s1_ids = read_ids(s1_file)
    s2_ids = read_ids(s2_file)
    s3_ids = read_ids(s3_file)
    s23_ids = s2_ids | s3_ids
    return s1_ids, s23_ids


def parse_tsv_file(fpath: str, expected_header_cols: list):
    """Parses a submission TSV and returns header and rows as dict."""
    issues = []
    rows = {}
    row_count = 0
    duplicate_s1 = set()

    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header is None:
            return None, None, [f"{fpath} is empty."]
        
        if header != expected_header_cols:
            issues.append(
                f"{fpath}: Invalid header. Expected {expected_header_cols}, got {header}"
            )
            return None, None, issues

        for line_no, row in enumerate(reader, start=2):
            row_count += 1
            if len(row) == 0:
                continue
            if len(row) == 1:
                # Could be S1 entity with empty list
                s1_id = row[0].strip()
                ids_str = ""
            elif len(row) == 2:
                s1_id = row[0].strip()
                ids_str = row[1].strip()
            else:
                issues.append(f"{fpath}: Line {line_no} has {len(row)} tab-separated fields, expected 2.")
                continue

            if s1_id in rows:
                duplicate_s1.add(s1_id)
            rows[s1_id] = ids_str

    if duplicate_s1:
        issues.append(
            f"{fpath}: Found {len(duplicate_s1)} duplicate source1_entity_id entries (e.g. {list(duplicate_s1)[:3]})."
        )

    return header, rows, issues


def validate(matching_file: str, candidate_file: str, test_dir: str):
    issues = []

    # Check files exist
    if not os.path.exists(matching_file):
        issues.append(f"Matching results file not found: {matching_file}")
    if not os.path.exists(candidate_file):
        issues.append(f"Candidate pairs file not found: {candidate_file}")
    if not os.path.exists(test_dir):
        issues.append(f"Test directory not found: {test_dir}")

    if issues:
        return issues

    try:
        test_s1_ids, test_s23_ids = load_test_entity_ids(test_dir)
    except Exception as e:
        issues.append(f"Error loading test entity IDs: {e}")
        return issues

    # Parse matching
    m_header, m_rows, m_issues = parse_tsv_file(
        matching_file, ["source1_entity_id", "matched_entity_ids"]
    )
    issues.extend(m_issues)

    # Parse candidate
    c_header, c_rows, c_issues = parse_tsv_file(
        candidate_file, ["source1_entity_id", "candidate_entity_ids"]
    )
    issues.extend(c_issues)

    if m_rows is None or c_rows is None:
        return issues

    # 1. Check entity coverage
    m_s1_ids = set(m_rows.keys())
    c_s1_ids = set(c_rows.keys())

    missing_in_m = test_s1_ids - m_s1_ids
    if missing_in_m:
        issues.append(
            f"matching_results.tsv is missing {len(missing_in_m)} Source 1 test entities (e.g. {list(missing_in_m)[:3]})."
        )

    extra_in_m = m_s1_ids - test_s1_ids
    if extra_in_m:
        issues.append(
            f"matching_results.tsv contains {len(extra_in_m)} unknown S1 entities not in test set (e.g. {list(extra_in_m)[:3]})."
        )

    missing_in_c = test_s1_ids - c_s1_ids
    if missing_in_c:
        issues.append(
            f"candidate_pairs.tsv is missing {len(missing_in_c)} Source 1 test entities (e.g. {list(missing_in_c)[:3]})."
        )

    extra_in_c = c_s1_ids - test_s1_ids
    if extra_in_c:
        issues.append(
            f"candidate_pairs.tsv contains {len(extra_in_c)} unknown S1 entities not in test set (e.g. {list(extra_in_c)[:3]})."
        )

    # 2. Check candidate contents and ID validity
    candidate_map = {}
    for s1_id, c_str in c_rows.items():
        if not c_str:
            candidate_map[s1_id] = set()
            continue
        c_list = [x.strip() for x in c_str.split(",") if x.strip()]
        if len(c_list) != len(set(c_list)):
            issues.append(f"candidate_pairs.tsv: Duplicate candidate IDs for entity {s1_id}")
        for cid in c_list:
            if cid.startswith("S1-"):
                issues.append(f"candidate_pairs.tsv: S1 self-match found for {s1_id}: {cid}")
            if cid not in test_s23_ids:
                issues.append(f"candidate_pairs.tsv: Unknown candidate ID {cid} for {s1_id}")
        candidate_map[s1_id] = set(c_list)

    # 3. Check matching contents, ID validity, and subset constraint
    for s1_id, m_str in m_rows.items():
        if not m_str:
            continue
        m_list = [x.strip() for x in m_str.split(",") if x.strip()]
        if len(m_list) != len(set(m_list)):
            issues.append(f"matching_results.tsv: Duplicate matched IDs for entity {s1_id}")
        for mid in m_list:
            if mid.startswith("S1-"):
                issues.append(f"matching_results.tsv: S1 self-match found for {s1_id}: {mid}")
            if mid not in test_s23_ids:
                issues.append(f"matching_results.tsv: Unknown matched ID {mid} for {s1_id}")

        c_set = candidate_map.get(s1_id, set())
        not_in_c = set(m_list) - c_set
        if not_in_c:
            issues.append(
                f"matching_results.tsv: Entity {s1_id} has matched IDs not present in candidate_pairs.tsv: {not_in_c}"
            )

    return issues


def main():
    parser = argparse.ArgumentParser(description="Validate ER challenge submission files.")
    parser.add_argument("--matching", required=True, help="Path to matching_results.tsv")
    parser.add_argument("--candidate", required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", required=True, help="Path to directory containing test_source1/2/3.tsv")
    args = parser.parse_args()

    issues = validate(args.matching, args.candidate, args.test_dir)
    if not issues:
        print("PASS")
        sys.exit(0)
    else:
        print(f"FAILED with {len(issues)} issue(s):")
        for i, issue in enumerate(issues[:30], 1):
            print(f"  {i}. {issue}")
        if len(issues) > 30:
            print(f"  ... and {len(issues) - 30} more.")
        sys.exit(1)


if __name__ == "__main__":
    main()
