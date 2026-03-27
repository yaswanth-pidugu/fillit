from __future__ import annotations

import re

from app.utils.constants import NETWORK_VALID_LENGTHS, PAN_MAX_LENGTH, PAN_MIN_LENGTH


def normalize_card_number(card_number: str) -> str:
    """Keep only digits from OCR output."""
    return re.sub(r"\D", "", card_number or "")


def luhn_check(card_number: str) -> bool:
    """Validate a PAN using the Luhn checksum algorithm."""
    pan = normalize_card_number(card_number)
    if len(pan) < PAN_MIN_LENGTH or len(pan) > PAN_MAX_LENGTH:
        return False

    total = 0
    parity = len(pan) % 2
    for index, char in enumerate(pan):
        digit = int(char)
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def length_valid_for_network(card_number: str, network: str) -> bool:
    pan = normalize_card_number(card_number)
    valid_lengths = NETWORK_VALID_LENGTHS.get(network.upper())
    if not valid_lengths:
        return PAN_MIN_LENGTH <= len(pan) <= PAN_MAX_LENGTH
    return len(pan) in valid_lengths


def is_strictly_valid_card_number(card_number: str, network: str) -> bool:
    pan = normalize_card_number(card_number)
    return (
        PAN_MIN_LENGTH <= len(pan) <= PAN_MAX_LENGTH
        and luhn_check(pan)
        and length_valid_for_network(pan, network)
    )


def score_candidate(
    ocr_confidence: float,
    card_number: str,
    network: str,
    bank: str,
) -> float:
    """Score formula with deterministic validation bonuses and [0,1] clipping."""
    bonus = 0.0
    if luhn_check(card_number):
        bonus += 0.25
    if length_valid_for_network(card_number, network):
        bonus += 0.15
    if network != "Unknown":
        bonus += 0.10
    if bank != "Unknown Bank":
        bonus += 0.05

    return max(0.0, min(1.0, float(ocr_confidence) + bonus))

