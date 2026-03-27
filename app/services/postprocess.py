from __future__ import annotations

import re
from datetime import datetime
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from app.services.bin_detector import bin_detector
from app.services.ocr_engine import OCRTextBlock
from app.services.validator import (
    is_strictly_valid_card_number,
    normalize_card_number,
    score_candidate,
)
from app.utils.constants import (
    CONFIDENCE_REJECT_THRESHOLD,
    NAME_STOPWORDS,
    UNKNOWN_BANK,
    UNKNOWN_VALUE,
)

PAN_PATTERN = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
EXPIRY_PATTERN = re.compile(r"(?<!\d)(0[1-9]|1[0-2])[ /-]?(\d{2}|\d{4})(?!\d)")


@dataclass
class CandidatePAN:
    card_number: str
    ocr_confidence: float
    network: str
    bank: str
    score: float


@dataclass
class ExtractedCardData:
    card_number: str
    expiry_date: str
    cardholder_name: str
    network_type: str
    bank_name: str
    confidence: float


def _known_field_count(data: ExtractedCardData) -> int:
    count = 0
    if data.card_number != UNKNOWN_VALUE:
        count += 3
    if data.expiry_date != UNKNOWN_VALUE:
        count += 1
    if data.cardholder_name != UNKNOWN_VALUE:
        count += 1
    if data.network_type != UNKNOWN_VALUE:
        count += 1
    if data.bank_name != UNKNOWN_BANK:
        count += 1
    return count


def select_best_extraction(
    candidates: Sequence[Tuple[str, ExtractedCardData, List[OCRTextBlock]]],
) -> Tuple[str, ExtractedCardData, List[OCRTextBlock]]:
    """Pick the strongest pass using field completeness, confidence, and OCR signal."""
    if not candidates:
        empty = ExtractedCardData(
            card_number=UNKNOWN_VALUE,
            expiry_date=UNKNOWN_VALUE,
            cardholder_name=UNKNOWN_VALUE,
            network_type=UNKNOWN_VALUE,
            bank_name=UNKNOWN_BANK,
            confidence=0.0,
        )
        return "none", empty, []

    def rank(item: Tuple[str, ExtractedCardData, List[OCRTextBlock]]) -> tuple[float, int, float]:
        _, data, blocks = item
        return (
            float(_known_field_count(data)),
            len(blocks),
            float(data.confidence),
        )

    return max(candidates, key=rank)


def _average_confidence(blocks: Iterable[OCRTextBlock]) -> float:
    values = [block.confidence for block in blocks]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def extract_card_number_candidates(blocks: List[OCRTextBlock]) -> List[CandidatePAN]:
    candidates: List[CandidatePAN] = []

    for block in blocks:
        for raw_match in PAN_PATTERN.findall(block.text):
            pan = normalize_card_number(raw_match)
            if not 13 <= len(pan) <= 19:
                continue
            network, bank = bin_detector.identify(pan)
            candidate = CandidatePAN(
                card_number=pan,
                ocr_confidence=block.confidence,
                network=network,
                bank=bank,
                score=score_candidate(block.confidence, pan, network, bank),
            )
            candidates.append(candidate)

    # OCR occasionally splits PANs across blocks, so also inspect merged text.
    merged_text = " ".join(block.text for block in blocks)
    merged_conf = _average_confidence(blocks)
    for raw_match in PAN_PATTERN.findall(merged_text):
        pan = normalize_card_number(raw_match)
        if not 13 <= len(pan) <= 19:
            continue
        network, bank = bin_detector.identify(pan)
        candidates.append(
            CandidatePAN(
                card_number=pan,
                ocr_confidence=merged_conf,
                network=network,
                bank=bank,
                score=score_candidate(merged_conf, pan, network, bank),
            )
        )

    unique: dict[str, CandidatePAN] = {}
    for candidate in candidates:
        existing = unique.get(candidate.card_number)
        if existing is None or candidate.score > existing.score:
            unique[candidate.card_number] = candidate

    return sorted(unique.values(), key=lambda item: item.score, reverse=True)


def extract_expiry_date(blocks: List[OCRTextBlock]) -> str:
    now = datetime.utcnow()
    current_yy = now.year % 100
    current_month = now.month

    candidates: List[tuple[float, str]] = []
    for block in blocks:
        for match in EXPIRY_PATTERN.finditer(block.text):
            month, year = match.group(1), match.group(2)
            yy = int(year[-2:])
            mm = int(month)

            # Accept only realistic validity windows to avoid noisy OCR hits.
            if yy < current_yy or yy > current_yy + 20:
                continue
            if yy == current_yy and mm < current_month:
                continue

            candidates.append((block.confidence, f"{month}/{yy:02d}"))

    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    return UNKNOWN_VALUE


def extract_cardholder_name(blocks: List[OCRTextBlock], extra_stopwords: Iterable[str] | None = None) -> str:
    stopwords = set(NAME_STOPWORDS)
    stopwords.update({token.upper() for token in (extra_stopwords or []) if token})

    candidates: List[str] = []
    for block in blocks:
        text = re.sub(r"[^A-Za-z\s.-]", " ", block.text).upper()
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 4:
            continue
        words = [word for word in text.split() if word.isalpha()]
        if len(words) < 2:
            continue

        filtered = [word for word in words if word not in stopwords]
        if len(filtered) < 2:
            continue

        candidate = " ".join(filtered)
        if any(stopword in candidate for stopword in stopwords):
            continue
        candidates.append(candidate)

    if not candidates:
        return UNKNOWN_VALUE
    return max(candidates, key=len)


def parse_ocr_blocks(blocks: List[OCRTextBlock]) -> ExtractedCardData:
    pan_candidates = extract_card_number_candidates(blocks)

    valid_candidates = [
        c for c in pan_candidates if is_strictly_valid_card_number(c.card_number, c.network)
    ]
    chosen = valid_candidates[0] if valid_candidates else None

    if chosen and chosen.score < CONFIDENCE_REJECT_THRESHOLD:
        chosen = None

    network = chosen.network if chosen else UNKNOWN_VALUE
    bank = chosen.bank if chosen else UNKNOWN_BANK
    card_number = chosen.card_number if chosen else UNKNOWN_VALUE

    stopwords = [network, bank]
    expiry = extract_expiry_date(blocks)
    name = extract_cardholder_name(blocks, stopwords)

    if chosen is not None:
        confidence = chosen.score
    else:
        confidence = min(0.5, _average_confidence(blocks))

    return ExtractedCardData(
        card_number=card_number,
        expiry_date=expiry,
        cardholder_name=name,
        network_type=network,
        bank_name=bank,
        confidence=round(confidence, 4),
    )


