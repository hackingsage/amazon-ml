"""
test_components.py - Comprehensive unit tests for normalization, blocking, features, and scoring.
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from code.business_entity_resolution.src.normalization import (
    normalize_business_name,
    normalize_address,
    normalize_country,
    normalize_text,
    get_name_tokens_for_blocking,
    get_combined_text,
    unicode_normalize,
    strip_legal_suffixes,
    normalize_legal_suffixes,
)
from code.business_entity_resolution.src.scoring import (
    f_beta_score,
    macro_f_beta,
)
from code.business_entity_resolution.src.features import (
    levenshtein_distance,
    levenshtein_ratio,
    jaro_similarity,
    jaro_winkler_similarity,
    jaccard_similarity,
    overlap_coefficient,
    compute_pair_features,
)


class TestNormalization(unittest.TestCase):
    def test_unicode_and_accents(self):
        self.assertEqual(unicode_normalize("Société Générale"), "societe generale")
        self.assertEqual(unicode_normalize("Crédit Agricole"), "credit agricole")
        self.assertEqual(normalize_text("Café & Résumé"), "cafe and resume")

    def test_legal_suffixes(self):
        # Canonicalization
        self.assertIn("corp", normalize_business_name("Acme Corporation"))
        self.assertIn("pvt", normalize_business_name("Infosys Private Limited"))
        self.assertIn("ltd", normalize_business_name("Infosys Private Limited"))
        self.assertIn("co", normalize_business_name("General Electric Company"))
        self.assertIn("inc", normalize_business_name("Apple Incorporated"))
        self.assertIn("llc", normalize_business_name("Amazon Services LLC"))
        # French suffixes
        self.assertIn("sarl", normalize_business_name("Boulangerie SARL"))
        self.assertIn("sas", normalize_business_name("Logistics SAS"))

    def test_suffix_stripping_for_blocking(self):
        tokens = get_name_tokens_for_blocking("Tata Consultancy Services Private Limited")
        self.assertIn("tata", tokens)
        self.assertIn("consultancy", tokens)
        # Should strip pvt, ltd, and svc
        self.assertNotIn("pvt", tokens)
        self.assertNotIn("ltd", tokens)
        self.assertNotIn("svc", tokens)

    def test_address_abbreviations(self):
        addr1 = normalize_address("123 Main Street, Suite 400, North Highway")
        self.assertEqual(addr1, "123 main st ste 400 n hwy")
        
        # Indian terms
        addr2 = normalize_address("Near SBI ATM, Sector 15, Gandhi Nagar")
        self.assertIn("nr", addr2)
        self.assertIn("sec", addr2)
        self.assertIn("nagar", addr2)

        # French terms
        addr3 = normalize_address("15 Rue de la Paix, Boulevard Saint-Germain")
        self.assertIn("rue", addr3)
        self.assertIn("blvd", addr3)

    def test_country_normalization_open_set(self):
        # Must handle any country string without hardcoded rejection
        self.assertEqual(normalize_country("  US  "), "us")
        self.assertEqual(normalize_country("India"), "india")
        self.assertEqual(normalize_country("France"), "france")
        self.assertEqual(normalize_country("Germany"), "germany")
        self.assertEqual(normalize_country("Brazil"), "brazil")


class TestScoring(unittest.TestCase):
    def test_problem_statement_example(self):
        # S1-00001 predicts [S2-00047, S2-00193, S3-00812]
        # Ground truth is [S2-00047, S3-00812]
        # Precision = 2/3, Recall = 1.0
        # F_0.5 = (1.25 * 0.6667 * 1.0) / (0.25 * 0.6667 + 1.0) = 0.714
        pred = {"S2-00047", "S2-00193", "S3-00812"}
        truth = {"S2-00047", "S3-00812"}
        score = f_beta_score(pred, truth, beta=0.5)
        self.assertAlmostEqual(score, 0.7142857, places=3)

    def test_singleton_scoring(self):
        # True singleton: truth is empty
        # If predicted is empty -> 1.0
        self.assertEqual(f_beta_score(set(), set(), beta=0.5), 1.0)
        # If predicted is non-empty -> 0.0 (false merge)
        self.assertEqual(f_beta_score({"S2-00001"}, set(), beta=0.5), 0.0)

    def test_non_singleton_missed(self):
        # True match exists, predicted empty -> 0.0
        self.assertEqual(f_beta_score(set(), {"S2-00001"}, beta=0.5), 0.0)

    def test_perfect_match(self):
        pred = {"S2-00001", "S3-00002"}
        truth = {"S2-00001", "S3-00002"}
        self.assertEqual(f_beta_score(pred, truth, beta=0.5), 1.0)

    def test_macro_averaging(self):
        # Entity 1: perfect match (1.0)
        # Entity 2: correct singleton (1.0)
        # Entity 3: false merge on singleton (0.0)
        # Entity 4: partial match (0.714)
        preds = {
            "S1-1": {"S2-1"},
            "S1-2": set(),
            "S1-3": {"S2-3"},
            "S1-4": {"S2-4a", "S2-4b", "S3-4"},
        }
        gt = {
            "S1-1": {"S2-1"},
            "S1-2": set(),
            "S1-3": set(),
            "S1-4": {"S2-4a", "S3-4"},
        }
        macro, per_entity = macro_f_beta(preds, gt, beta=0.5)
        expected = (1.0 + 1.0 + 0.0 + (1.25 * (2/3) * 1.0) / (0.25 * (2/3) + 1.0)) / 4.0
        self.assertAlmostEqual(macro, expected, places=4)


class TestFeatures(unittest.TestCase):
    def test_similarity_metrics(self):
        self.assertEqual(levenshtein_distance("kitten", "sitting"), 3)
        self.assertAlmostEqual(levenshtein_ratio("kitten", "kitten"), 1.0)
        self.assertGreater(jaro_winkler_similarity("dixon", "dicksonx"), 0.8)
        self.assertEqual(jaccard_similarity({"a", "b"}, {"a", "b"}), 1.0)
        self.assertEqual(overlap_coefficient({"a", "b"}, {"a", "c", "d"}), 0.5)

    def test_pair_feature_generation(self):
        s1 = {
            "entity_id": "S1-100",
            "business_name": "Acme Industrial Technologies Ltd",
            "business_address": "100 Market St, Suite 500, San Francisco",
            "country": "US",
        }
        s2 = {
            "entity_id": "S2-200",
            "business_name": "Acme Industrial Tech Inc",
            "business_address": "100 Market Street, Ste 500, SF",
            "country": "US",
        }
        feats = compute_pair_features(s1, s2)
        self.assertIn("name_jaro_winkler", feats)
        self.assertIn("addr_jaro_winkler", feats)
        self.assertIn("country_exact_match", feats)
        self.assertEqual(feats["country_exact_match"], 1.0)
        self.assertGreater(feats["name_jaro_winkler"], 0.85)


if __name__ == "__main__":
    unittest.main()
