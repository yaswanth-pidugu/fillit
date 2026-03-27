from __future__ import annotations

from typing import Dict, Set, Tuple

UNKNOWN_VALUE = "Unknown"
UNKNOWN_BANK = "Unknown Bank"

OCR_LANG = "en"
OCR_USE_ANGLE_CLS = True
OCR_USE_GPU = False

PAN_MIN_LENGTH = 13
PAN_MAX_LENGTH = 19
CONFIDENCE_REJECT_THRESHOLD = 0.55

NETWORK_RULES = {
    "VISA": [("prefix", "4")],
    "MASTERCARD": [
        ("range", ("51", "55")),
        ("range", ("2221", "2720")),
    ],
    "AMEX": [("prefix", "34"), ("prefix", "37")],
    "DISCOVER": [
        ("prefix", "6011"),
        ("prefix", "65"),
        ("range", ("644", "649")),
    ],
    "JCB": [("range", ("3528", "3589"))],
    "DINERS": [
        ("range", ("300", "305")),
        ("prefix", "36"),
        ("prefix", "38"),
    ],
}

NETWORK_VALID_LENGTHS: Dict[str, Set[int]] = {
    "VISA": {13, 16, 19},
    "MASTERCARD": {16},
    "AMEX": {15},
    "DISCOVER": {16, 19},
    "JCB": {16, 19},
    "DINERS": {14},
}

DEFAULT_BIN_TABLE: Dict[str, Tuple[str, str]] = {
    "411111": ("JPMorgan Chase", "VISA"),
    "401288": ("Wells Fargo", "VISA"),
    "555555": ("Bank of America", "MASTERCARD"),
    "520082": ("Citi", "MASTERCARD"),
    "378282": ("American Express", "AMEX"),
    "601111": ("Discover Financial", "DISCOVER"),
}

NAME_STOPWORDS = {
    "VALID",
    "THRU",
    "VALID THRU",
    "VISA",
    "MASTERCARD",
    "AMEX",
    "DISCOVER",
    "CARD",
    "DEBIT",
    "CREDIT",
    "BANK",
    "PLATINUM",
    "GOLD",
    "WORLD",
    "ELECTRON",
    "PAY",
}

