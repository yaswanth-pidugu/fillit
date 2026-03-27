from __future__ import annotations

from typing import Dict, Mapping, Tuple

from app.services.validator import normalize_card_number
from app.utils.constants import DEFAULT_BIN_TABLE, UNKNOWN_BANK, UNKNOWN_VALUE

BinRecord = Tuple[str, str]


class BinDetector:
    """BIN and network detector with strict prefix/range rules."""

    def __init__(self, bin_table: Mapping[str, BinRecord] | None = None) -> None:
        self._bin_table: Dict[str, BinRecord] = dict(bin_table or DEFAULT_BIN_TABLE)

    def update_table(self, entries: Mapping[str, BinRecord]) -> None:
        self._bin_table.update(entries)

    def detect_network(self, card_number: str) -> str:
        pan = normalize_card_number(card_number)
        if not pan:
            return UNKNOWN_VALUE

        first_one = int(pan[:1]) if len(pan) >= 1 else -1
        first_two = int(pan[:2]) if len(pan) >= 2 else -1
        first_three = int(pan[:3]) if len(pan) >= 3 else -1
        first_four = int(pan[:4]) if len(pan) >= 4 else -1

        if first_one == 4:
            return "VISA"
        if 51 <= first_two <= 55 or 2221 <= first_four <= 2720:
            return "MASTERCARD"
        if first_two in {34, 37}:
            return "AMEX"
        if pan.startswith("6011") or pan.startswith("65") or 644 <= first_three <= 649:
            return "DISCOVER"
        if 3528 <= first_four <= 3589:
            return "JCB"
        if 300 <= first_three <= 305 or first_two in {36, 38}:
            return "DINERS"

        return UNKNOWN_VALUE

    def detect_bank(self, card_number: str) -> str:
        pan = normalize_card_number(card_number)
        if len(pan) < 6:
            return UNKNOWN_BANK
        entry = self._bin_table.get(pan[:6])
        if not entry:
            return UNKNOWN_BANK
        return entry[0]

    def identify(self, card_number: str) -> Tuple[str, str]:
        pan = normalize_card_number(card_number)
        if len(pan) < 6:
            return UNKNOWN_VALUE, UNKNOWN_BANK

        network = self.detect_network(pan)
        bank = self.detect_bank(pan)

        # If BIN contains better network metadata, use it.
        bin_entry = self._bin_table.get(pan[:6])
        if bin_entry and bin_entry[1]:
            network = bin_entry[1]

        return network, bank


bin_detector = BinDetector()

