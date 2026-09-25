# Methodology Write-Up: Business Entity Resolution

## 1. Executive Summary

This solution presents a scalable, high-precision two-stage pipeline for matching noisy business entity records across three disparate data sources (Source 1 reference, Sources 2 and 3 un-deduplicated noisy query sources).

The evaluation metric is **Macro-Averaged F₀.₅** across all Source 1 entities, which weights precision twice as heavily as recall and places high reward on correctly predicting singletons (1.0 points for correctly identifying an entity with 0 matches, 0.0 for any false merge). Our pipeline is engineered from the ground up to maximize precision while preserving high candidate recall ceiling.

---

## 2. Architecture Overview

The system consists of two primary stages followed by calibrated decision thresholding:

```
┌─────────────────────────────────────────────────────────────┐
│                    Stage 1: Normalization                   │
│  - Unicode decomposition & diacritic stripping              │
│  - Multi-jurisdiction legal suffix canonicalization         │
│  - Landmark & street abbreviation normalization             │
│  - Open-set country standardization                         │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              Stage 2: Hybrid Union Blocking                 │
│  - Token Inverted Index (Name tokens + Country)             │
│  - Sublinear TF-IDF Character 3/4-gram Cosine Top-K         │
│  - Union merge: Guarantees high recall ceiling              │
│  → candidate_pairs.tsv                                      │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│            Stage 3: Feature Engineering (~30 dims)          │
│  - String metrics: Jaro-Winkler, Levenshtein, N-gram Jaccard│
│  - Token metrics: Jaccard, Overlap coefficient, Count ratio │
│  - Suffix-stripped name similarity                          │
│  - Address & landmark similarity                            │
│  - Cross-source indicator & open country match flags        │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│          Stage 4: Gradient-Boosted Trees (LightGBM)         │
│  - Binary classification with class-imbalance weighting     │
│  - F₀.₅-specific threshold sweep on validation holdout      │
│  - High decision threshold to suppress false merges         │
│  - Fallback logic to protect singleton credit               │
│  → matching_results.tsv                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Stage 1: Candidate Generation / Blocking Strategy

### The Recall Ceiling Challenge
Entity resolution over $N_1 \times (N_2 + N_3)$ pairs exhibits quadratic complexity ($O(N^2)$). Naive pairwise scoring is intractable. However, any true match pruned during blocking can never be recovered. Therefore, blocking must maximize the **recall ceiling** while maintaining a high **reduction ratio** (>99%).

### The Union Approach
We combine two complementary blocking mechanisms:

1. **Token Inverted Index (High Selectivity)**:
   - Strips legal suffixes (e.g., *Corp, Pvt, Ltd, LLC, SARL, SAS*) to isolate core brand tokens.
   - Indexes records under compound keys `country|token` as well as global `*|token` keys.
   - Frequency capping filters out ubiquitous stopword-like tokens (blocks > 500 records).
   - Catches canonical, reordered, and exact-brand variations instantaneously ($O(1)$ lookup).

2. **Sublinear TF-IDF Character N-gram Top-K (High Fuzzy Recall)**:
   - Employs character 3-gram and 4-gram TF-IDF vectorization across combined normalized name, address, and country fields.
   - Handles severe typos, phonetic variations, and transliteration differences where token overlap is zero.
   - Performs sparse matrix cosine multiplication (`batch_dot`) to retrieve the top $K=30$ candidate matches per Source 1 entity.

3. **Union Merge**:
   - The candidate set is the union: $\mathcal{C}(S_1) = \mathcal{C}_{\text{token}}(S_1) \cup \mathcal{C}_{\text{tfidf}}(S_1)$.
   - All pairs fed to the matching model are persisted to `candidate_pairs.tsv`.

---

## 4. Stage 2: Feature Engineering & Model Architecture

### Feature Engineering (31 Features)
For each candidate pair $(s_1, s_{23})$, we compute 31 similarity features:

- **Name Similarity**:
  - `name_jaro_winkler`: Captures prefix-biased similarity.
  - `name_levenshtein_ratio`: Normalized edit distance.
  - `name_char_3gram_jaccard` & `name_char_4gram_jaccard`: Fine-grained sub-token overlap.
  - `name_ns_jaro_winkler` & `name_ns_levenshtein_ratio`: Similarity calculated after stripping legal suffixes.
  - `name_exact_match` & `name_ns_exact_match`: Binary flags for exact raw and suffix-free equality.
  - `name_token_jaccard` & `name_token_overlap`: Token set intersection over union and overlap coefficient.
  - `name_shared_token_count` & `name_ns_shared_token_count`: Absolute counts of shared tokens.
  - `name_containment`: Substring containment indicator for acronyms/abbreviations.
  - `name_length_ratio` & `name_token_count_ratio`: Structural dimension consistency.
- **Address & Landmark Similarity**:
  - `addr_jaro_winkler` & `addr_levenshtein_ratio`: Address string metrics.
  - `addr_char_3gram_jaccard`: Handles localized street formatting variations.
  - `addr_token_jaccard`, `addr_token_overlap`, `addr_shared_token_count`: Token overlap metrics.
  - `addr_exact_match` & `addr_length_ratio`.
- **Country & Cross-Source Compatibility**:
  - `country_exact_match`: 1.0 if normalized country strings match exactly, 0.0 otherwise.
  - `country_either_missing`: Handles missing country attributes without failure.
  - `is_source3`: Distinguishes between Source 2 and Source 3 characteristics.
- **Holistic Combined Similarity**:
  - `combined_jaro_winkler`, `combined_char_3gram_jaccard`, `combined_token_jaccard` across concatenated name and address.

### Model Architecture: LightGBM
- **Model**: LightGBM Gradient Boosted Decision Trees (GBDT).
- **License**: MIT License (fully compliant with the open-license constraint).
- **Parameter Size**: Compact binary tree model (< 200 trees, ~5 MB), well under the 8-billion parameter ceiling.
- **Class Imbalance**: Utilizes `scale_pos_weight = N_neg / N_pos` during training to account for the heavy imbalance between candidate negatives and true positives.

---

## 5. Metric Alignment: F₀.₅ Optimization & Singleton Strategy

The competition uses macro-averaged $F_{0.5}$:
$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

### Impact of Precision Weighting
- In $F_1$, precision and recall are weighted equally. In $F_{0.5}$, **precision is weighted twice as heavily as recall**.
- Merging two distinct entities (a false positive) degrades precision and incurs a severe penalty.
- Missing a weak, ambiguous link (a false negative) incurs a much smaller penalty.

### The Singleton Advantage
- In macro-averaging, each Source 1 entity contributes an independent score in $[0, 1]$.
- A true singleton (an entity with no matches in S2 or S3) scores:
  - **1.0** if predicted as an empty list (correctly recognized as singleton).
  - **0.0** if even a single false match is predicted.
- Therefore, predicting matches too aggressively destroys the macro-average on all singleton entities.

### Threshold Sweep Strategy
Instead of using a default 0.5 probability cutoff, our pipeline performs a grid sweep over thresholds $\theta \in [0.10, 0.95]$ with step $0.02$ on a held-out validation split. The threshold that strictly maximizes the competition's macro-averaged $F_{0.5}$ formula is selected and saved to `threshold.txt`.

---

## 6. Generalization & Fair Play Compliance

1. **Unseen Country Generalization (France)**:
   - No country labels are hardcoded, filtered, or one-hot encoded to `{US, India}`.
   - The normalization dictionary includes common French legal forms (*SARL, SAS, SA, Société, Cie*) and street terms (*Rue, Boulevard, Allee, Chemin, Cedex*).
   - Character n-gram blocking and country-equality matching dynamically adapt to any unseen country string.
2. **Zero External API / Data Lookup**:
   - Strictly self-contained: no geocoding APIs, no external postal code lookups, no web search, no commercial ER services.
   - All string similarity metrics and normalizers are computed entirely in local memory using standard Python libraries and scikit-learn.
3. **Format & Rule Adherence**:
   - Tab-separated TSVs with `sep="\t"`.
   - Header contracts strictly match `source1_entity_id`, `matched_entity_ids` / `candidate_entity_ids`.
   - Verified with the standalone `utils/validate_submission.py` script.
