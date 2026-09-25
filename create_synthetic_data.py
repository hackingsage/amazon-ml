"""
create_synthetic_data.py - Generates realistic synthetic benchmark dataset.

Used to test and validate the end-to-end pipeline before actual competition
data is dropped in.
"""

import os
import pandas as pd


def generate_synthetic_data():
    os.makedirs("dataset/train", exist_ok=True)
    os.makedirs("dataset/test", exist_ok=True)

    # ---------------------------------------------------------
    # TRAIN SET (US and India)
    # ---------------------------------------------------------
    train_s1 = [
        # Entity 1 (India): matches S2-001 and S3-001
        {
            "entity_id": "S1-001",
            "business_name": "Infosys Technologies Private Limited",
            "business_address": "Electronics City, Hosur Road, Bangalore, Karnataka 560100",
            "country": "India",
        },
        # Entity 2 (US): matches S2-002
        {
            "entity_id": "S1-002",
            "business_name": "Amazon Web Services Inc",
            "business_address": "410 Terry Avenue North, Seattle, WA 98109",
            "country": "US",
        },
        # Entity 3 (India): matches S3-003
        {
            "entity_id": "S1-003",
            "business_name": "Tata Consultancy Services Limited",
            "business_address": "Nirmal Building, 9th Floor, Nariman Point, Mumbai 400021",
            "country": "India",
        },
        # Entity 4 (US): Singleton (No match)
        {
            "entity_id": "S1-004",
            "business_name": "Blue River Bakery & Cafe LLC",
            "business_address": "742 Evergreen Terrace, Springfield, OR 97477",
            "country": "US",
        },
        # Entity 5 (India): matches S2-005
        {
            "entity_id": "S1-005",
            "business_name": "State Bank of India Corporate Centre",
            "business_address": "State Bank Bhavan, Madame Cama Road, Nariman Point, Mumbai 400021",
            "country": "India",
        },
        # Entity 6 (US): matches S2-006 and S3-006
        {
            "entity_id": "S1-006",
            "business_name": "Microsoft Corporation",
            "business_address": "One Microsoft Way, Redmond, WA 98052",
            "country": "US",
        },
        # Entity 7 (India): Singleton (No match)
        {
            "entity_id": "S1-007",
            "business_name": "Sharma Sweet House & Catering",
            "business_address": "Near Clock Tower, Chandni Chowk, Delhi 110006",
            "country": "India",
        },
        # Entity 8 (US): Singleton (No match)
        {
            "entity_id": "S1-008",
            "business_name": "Pacific Horizon Logistics Group",
            "business_address": "1200 Harbor Boulevard, Suite 300, Long Beach, CA 90802",
            "country": "US",
        },
        # Entity 9 (India): matches S2-009
        {
            "entity_id": "S1-009",
            "business_name": "Reliance Industries Limited",
            "business_address": "Maker Chambers IV, 3rd Floor, 222 Nariman Point, Mumbai",
            "country": "India",
        },
        # Entity 10 (US): matches S3-010
        {
            "entity_id": "S1-010",
            "business_name": "Alphabet Google Headquarters",
            "business_address": "1600 Amphitheatre Parkway, Mountain View, CA 94043",
            "country": "US",
        },
    ]

    train_s2 = [
        {
            "entity_id": "S2-001",
            "business_name": "Infosys Tech Pvt Ltd",
            "business_address": "Hosur Rd, Electronic City, Bengaluru",
            "country": "India",
        },
        {
            "entity_id": "S2-002",
            "business_name": "AWS Amazon Services Incorporated",
            "business_address": "410 Terry Ave N, Seattle, Washington",
            "country": "US",
        },
        {
            "entity_id": "S2-005",
            "business_name": "SBI Corp Centre",
            "business_address": "Nr Madame Cama Rd, Nariman Pt, Mumbai",
            "country": "India",
        },
        {
            "entity_id": "S2-006",
            "business_name": "Microsoft Corp",
            "business_address": "1 Microsoft Way, Redmond, WA",
            "country": "US",
        },
        {
            "entity_id": "S2-009",
            "business_name": "Reliance Ind Ltd",
            "business_address": "Maker Chambers 4, Nariman Point, Mumbai",
            "country": "India",
        },
        # Distractor records in S2
        {
            "entity_id": "S2-991",
            "business_name": "Acme Hardware Store",
            "business_address": "100 Broadway, New York, NY",
            "country": "US",
        },
        {
            "entity_id": "S2-992",
            "business_name": "Patel Grocers & Spices",
            "business_address": "Sector 18, Noida, Uttar Pradesh",
            "country": "India",
        },
    ]

    train_s3 = [
        {
            "entity_id": "S3-001",
            "business_name": "Infosys Ltd",
            "business_address": "Electronics City Phase 1, Hosur Road, Bangalore",
            "country": "India",
        },
        {
            "entity_id": "S3-003",
            "business_name": "TCS Tata Consultancy Services Ltd",
            "business_address": "Nirmal Bldg, Nariman Point, Mumbai",
            "country": "India",
        },
        {
            "entity_id": "S3-006",
            "business_name": "Microsoft Corporation Inc",
            "business_address": "One Microsoft Way, Redmond, WA 98052",
            "country": "US",
        },
        {
            "entity_id": "S3-010",
            "business_name": "Google LLC (Alphabet)",
            "business_address": "1600 Amphitheatre Pkwy, Mountain View, California",
            "country": "US",
        },
        # Distractor records in S3
        {
            "entity_id": "S3-991",
            "business_name": "Pasadena Dental Clinic",
            "business_address": "450 Colorado Blvd, Pasadena, CA",
            "country": "US",
        },
        {
            "entity_id": "S3-992",
            "business_name": "Gupta Medical Hall",
            "business_address": "Main Market, Karol Bagh, New Delhi",
            "country": "India",
        },
    ]

    # Ground truth mapping
    ground_truth = [
        {"source1_entity_id": "S1-001", "matched_entity_ids": "S2-001,S3-001"},
        {"source1_entity_id": "S1-002", "matched_entity_ids": "S2-002"},
        {"source1_entity_id": "S1-003", "matched_entity_ids": "S3-003"},
        {"source1_entity_id": "S1-004", "matched_entity_ids": ""},  # Singleton
        {"source1_entity_id": "S1-005", "matched_entity_ids": "S2-005"},
        {"source1_entity_id": "S1-006", "matched_entity_ids": "S2-006,S3-006"},
        {"source1_entity_id": "S1-007", "matched_entity_ids": ""},  # Singleton
        {"source1_entity_id": "S1-008", "matched_entity_ids": ""},  # Singleton
        {"source1_entity_id": "S1-009", "matched_entity_ids": "S2-009"},
        {"source1_entity_id": "S1-010", "matched_entity_ids": "S3-010"},
    ]

    # ---------------------------------------------------------
    # TEST SET (US, India, and France)
    # ---------------------------------------------------------
    test_s1 = [
        # France entity
        {
            "entity_id": "S1-T01",
            "business_name": "Société Générale S.A.",
            "business_address": "29 Boulevard Haussmann, 75009 Paris",
            "country": "France",
        },
        # India entity
        {
            "entity_id": "S1-T02",
            "business_name": "Wipro Technologies Limited",
            "business_address": "Doddakannelli, Sarjapur Road, Bangalore 560035",
            "country": "India",
        },
        # US entity
        {
            "entity_id": "S1-T03",
            "business_name": "Apple Incorporated",
            "business_address": "One Apple Park Way, Cupertino, CA 95014",
            "country": "US",
        },
        # France singleton
        {
            "entity_id": "S1-T04",
            "business_name": "Boulangerie Patisserie Dupont SARL",
            "business_address": "12 Rue de la Paix, 69002 Lyon",
            "country": "France",
        },
        # US singleton
        {
            "entity_id": "S1-T05",
            "business_name": "Cascadia Timber Craft LLC",
            "business_address": "88 Pine St, Seattle, WA 98101",
            "country": "US",
        },
    ]

    test_s2 = [
        {
            "entity_id": "S2-T01",
            "business_name": "Societe Generale SA",
            "business_address": "29 Blvd Haussmann, Paris",
            "country": "France",
        },
        {
            "entity_id": "S2-T02",
            "business_name": "Wipro Tech Ltd",
            "business_address": "Sarjapur Rd, Doddakannelli, Bengaluru",
            "country": "India",
        },
        {
            "entity_id": "S2-T03",
            "business_name": "Apple Inc",
            "business_address": "1 Apple Park Way, Cupertino, California",
            "country": "US",
        },
        # Distractor
        {
            "entity_id": "S2-TD1",
            "business_name": "Bordeaux Wine Cellars SAS",
            "business_address": "5 Quai des Chartrons, Bordeaux",
            "country": "France",
        },
    ]

    test_s3 = [
        {
            "entity_id": "S3-T01",
            "business_name": "Société Générale",
            "business_address": "29 Boulevard Haussmann, Paris 75009",
            "country": "France",
        },
        {
            "entity_id": "S3-T03",
            "business_name": "Apple Computer Company",
            "business_address": "One Apple Park Way, Cupertino, CA",
            "country": "US",
        },
        # Distractor
        {
            "entity_id": "S3-TD1",
            "business_name": "Delhi Auto Works",
            "business_address": "Mayapuri Industrial Area, New Delhi",
            "country": "India",
        },
    ]

    # Save training files
    pd.DataFrame(train_s1).to_csv("dataset/train/train_source1.tsv", sep="\t", index=False)
    pd.DataFrame(train_s2).to_csv("dataset/train/train_source2.tsv", sep="\t", index=False)
    pd.DataFrame(train_s3).to_csv("dataset/train/train_source3.tsv", sep="\t", index=False)
    pd.DataFrame(ground_truth).to_csv("dataset/train/train_ground_truth.tsv", sep="\t", index=False)

    # Save test files
    pd.DataFrame(test_s1).to_csv("dataset/test/test_source1.tsv", sep="\t", index=False)
    pd.DataFrame(test_s2).to_csv("dataset/test/test_source2.tsv", sep="\t", index=False)
    pd.DataFrame(test_s3).to_csv("dataset/test/test_source3.tsv", sep="\t", index=False)

    print("Synthetic dataset created successfully in dataset/train and dataset/test.")


if __name__ == "__main__":
    generate_synthetic_data()
