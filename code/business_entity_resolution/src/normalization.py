"""
normalization.py — Reusable text normalization for business names and addresses.

Used by both blocking and matching feature stages. Contains:
  - Legal suffix / abbreviation dictionaries
  - Address abbreviation dictionaries
  - Unicode normalization, case folding, punctuation cleanup
  - Tokenization helpers

Does NOT hardcode country values or use external lookups.
"""

import re
import unicodedata
from typing import List


# ---------------------------------------------------------------------------
# Legal suffix / abbreviation mappings  (canonical form → variants)
# ---------------------------------------------------------------------------
# We store as variant → canonical so lookup is O(1).
# All keys are lowercase, no punctuation.

_LEGAL_SUFFIX_MAP_RAW = {
    # Corporation
    "corporation": "corp",
    "corp": "corp",
    "corpn": "corp",
    # Company
    "company": "co",
    "co": "co",
    # Incorporated
    "incorporated": "inc",
    "inc": "inc",
    "incorp": "inc",
    # Limited
    "limited": "ltd",
    "ltd": "ltd",
    "ltda": "ltd",  # Spanish/Portuguese variant
    # Private Limited
    "private": "pvt",
    "pvt": "pvt",
    "pte": "pvt",  # Singapore-style
    # Public Limited Company
    "plc": "plc",
    # Limited Liability
    "llc": "llc",
    "llp": "llp",
    # International
    "international": "intl",
    "intl": "intl",
    "interntional": "intl",  # common typo
    # Industries
    "industries": "ind",
    "ind": "ind",
    "indus": "ind",
    # Enterprise / Enterprises
    "enterprise": "ent",
    "enterprises": "ent",
    "ent": "ent",
    # Technologies / Technology
    "technologies": "tech",
    "technology": "tech",
    "tech": "tech",
    # Solutions
    "solutions": "sol",
    "solution": "sol",
    "sol": "sol",
    # Services
    "services": "svc",
    "service": "svc",
    "svc": "svc",
    "svcs": "svc",
    # Group
    "group": "grp",
    "grp": "grp",
    # Holdings
    "holdings": "hldg",
    "holding": "hldg",
    "hldg": "hldg",
    # Associates
    "associates": "assoc",
    "associate": "assoc",
    "assoc": "assoc",
    # Foundation
    "foundation": "fdn",
    "fdn": "fdn",
    # Partners
    "partners": "partners",
    "partner": "partners",
    # Trading
    "trading": "trdg",
    "trdg": "trdg",
    # Doing-Business-As
    "dba": "dba",
    # Société (French — test set includes France)
    "sarl": "sarl",
    "sas": "sas",
    "sa": "sa",
    "societe": "societe",
    "société": "societe",
    "cie": "cie",
    "compagnie": "cie",
    "etablissements": "ets",
    "ets": "ets",
    "freres": "freres",
    "frères": "freres",
}

# ---------------------------------------------------------------------------
# Address abbreviation mappings
# ---------------------------------------------------------------------------
_ADDRESS_ABBREV_MAP_RAW = {
    # Street types
    "street": "st",
    "st": "st",
    "str": "st",
    "road": "rd",
    "rd": "rd",
    "avenue": "ave",
    "ave": "ave",
    "av": "ave",
    "boulevard": "blvd",
    "blvd": "blvd",
    "drive": "dr",
    "dr": "dr",
    "lane": "ln",
    "ln": "ln",
    "place": "pl",
    "pl": "pl",
    "court": "ct",
    "ct": "ct",
    "circle": "cir",
    "cir": "cir",
    "terrace": "ter",
    "ter": "ter",
    "highway": "hwy",
    "hwy": "hwy",
    "parkway": "pkwy",
    "pkwy": "pkwy",
    "expressway": "expy",
    "expy": "expy",
    "square": "sq",
    "sq": "sq",
    "trail": "trl",
    "trl": "trl",
    # Directions
    "north": "n",
    "south": "s",
    "east": "e",
    "west": "w",
    "northeast": "ne",
    "northwest": "nw",
    "southeast": "se",
    "southwest": "sw",
    "n": "n",
    "s": "s",
    "e": "e",
    "w": "w",
    "ne": "ne",
    "nw": "nw",
    "se": "se",
    "sw": "sw",
    # Building / unit
    "apartment": "apt",
    "apt": "apt",
    "suite": "ste",
    "ste": "ste",
    "building": "bldg",
    "bldg": "bldg",
    "floor": "fl",
    "fl": "fl",
    "flr": "fl",
    "room": "rm",
    "rm": "rm",
    "unit": "unit",
    # Indian address terms
    "nagar": "nagar",
    "marg": "marg",
    "chowk": "chowk",
    "bazaar": "bazar",
    "bazar": "bazar",
    "gali": "gali",
    "mohalla": "mohalla",
    "sector": "sec",
    "sec": "sec",
    "phase": "ph",
    "ph": "ph",
    "block": "blk",
    "blk": "blk",
    "plot": "plot",
    "near": "nr",
    "nr": "nr",
    "opposite": "opp",
    "opp": "opp",
    "behind": "behind",
    # French address terms  (test set includes France)
    "rue": "rue",
    "avenue": "ave",
    "boulevard": "blvd",
    "place": "pl",
    "impasse": "imp",
    "imp": "imp",
    "passage": "pass",
    "pass": "pass",
    "allee": "allee",
    "allée": "allee",
    "chemin": "chemin",
    "route": "rte",
    "rte": "rte",
    "quartier": "quartier",
    "cedex": "cedex",
    # General
    "mount": "mt",
    "mt": "mt",
    "mountain": "mtn",
    "mtn": "mtn",
    "fort": "ft",
    "ft": "ft",
    "saint": "st",  # context-dependent but common
    "district": "dist",
    "dist": "dist",
    "post office": "po",
    "po": "po",
    "number": "no",
    "no": "no",
}

# ---------------------------------------------------------------------------
# Compiled lookup dicts (built once at import time)
# ---------------------------------------------------------------------------
LEGAL_SUFFIX_MAP: dict[str, str] = {k.lower(): v for k, v in _LEGAL_SUFFIX_MAP_RAW.items()}
ADDRESS_ABBREV_MAP: dict[str, str] = {k.lower(): v for k, v in _ADDRESS_ABBREV_MAP_RAW.items()}

# Set of all known legal suffix canonical forms — used to strip them from names
LEGAL_SUFFIX_CANONICALS: set[str] = set(LEGAL_SUFFIX_MAP.values())

# Regex: characters to strip (anything that isn't alphanumeric or whitespace)
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
# Regex: collapse multiple spaces
_MULTI_SPACE_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Core normalizers
# ---------------------------------------------------------------------------

def unicode_normalize(text: str) -> str:
    """NFC → NFKD → strip accents → lowercase."""
    if not text:
        return ""
    # Normalize unicode
    text = unicodedata.normalize("NFKD", text)
    # Strip combining marks (accents) — keeps base letters
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower()


def remove_punctuation(text: str) -> str:
    """Remove all punctuation, keeping alphanumeric and whitespace."""
    text = text.replace("&", " and ")  # & → and before stripping
    text = _PUNCT_RE.sub(" ", text)
    return _MULTI_SPACE_RE.sub(" ", text).strip()


def normalize_text(text: str) -> str:
    """Full normalization: unicode → punctuation → whitespace."""
    if not text or (isinstance(text, float)):
        return ""
    text = str(text)
    text = unicode_normalize(text)
    text = remove_punctuation(text)
    return text


def tokenize(text: str) -> List[str]:
    """Split normalized text into tokens."""
    return normalize_text(text).split()


def normalize_legal_suffixes(tokens: List[str]) -> List[str]:
    """Replace legal suffix tokens with canonical forms."""
    return [LEGAL_SUFFIX_MAP.get(t, t) for t in tokens]


def strip_legal_suffixes(tokens: List[str]) -> List[str]:
    """Remove all legal suffix tokens entirely — useful for blocking keys."""
    return [t for t in tokens if LEGAL_SUFFIX_MAP.get(t, t) not in LEGAL_SUFFIX_CANONICALS]


def normalize_address_abbrevs(tokens: List[str]) -> List[str]:
    """Replace address abbreviation tokens with canonical forms."""
    return [ADDRESS_ABBREV_MAP.get(t, t) for t in tokens]


# ---------------------------------------------------------------------------
# High-level field normalizers
# ---------------------------------------------------------------------------

def normalize_business_name(name: str, keep_suffixes: bool = True) -> str:
    """
    Full normalization of a business name.
    
    Args:
        name: Raw business name string.
        keep_suffixes: If True, canonicalize legal suffixes.
                       If False, strip them entirely (for blocking keys).
    Returns:
        Normalized name string with tokens joined by space.
    """
    tokens = tokenize(name)
    if keep_suffixes:
        tokens = normalize_legal_suffixes(tokens)
    else:
        tokens = strip_legal_suffixes(tokens)
    return " ".join(tokens)


def normalize_address(address: str) -> str:
    """
    Full normalization of a business address.
    
    Returns:
        Normalized address string with tokens joined by space.
    """
    tokens = tokenize(address)
    tokens = normalize_address_abbrevs(tokens)
    return " ".join(tokens)


def normalize_country(country: str) -> str:
    """
    Normalize country label — lowercase, stripped.
    No hardcoding of country values — treats as open set of strings.
    """
    if not country or (isinstance(country, float)):
        return ""
    return str(country).strip().lower()


def get_name_tokens_for_blocking(name: str) -> List[str]:
    """
    Get name tokens for blocking — suffixes stripped, each token individually.
    Returns deduplicated tokens sorted for consistency.
    """
    tokens = tokenize(name)
    tokens = strip_legal_suffixes(tokens)
    # Remove very short tokens (1 char) that cause spurious blocks
    tokens = [t for t in tokens if len(t) > 1]
    return sorted(set(tokens))


def get_combined_text(name: str, address: str, country: str) -> str:
    """
    Concatenate normalized name + address + country for TF-IDF blocking.
    """
    parts = []
    n = normalize_business_name(name, keep_suffixes=True)
    if n:
        parts.append(n)
    a = normalize_address(address)
    if a:
        parts.append(a)
    c = normalize_country(country)
    if c:
        parts.append(c)
    return " ".join(parts)
