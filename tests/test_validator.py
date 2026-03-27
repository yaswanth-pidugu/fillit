from app.services.validator import (
    length_valid_for_network,
    luhn_check,
    normalize_card_number,
)


def test_normalize_card_number() -> None:
    assert normalize_card_number("4111 1111-1111 1111") == "4111111111111111"


def test_luhn_check_valid_and_invalid() -> None:
    assert luhn_check("4111111111111111") is True
    assert luhn_check("4111111111111112") is False


def test_network_length_validation() -> None:
    assert length_valid_for_network("378282246310005", "AMEX") is True
    assert length_valid_for_network("3782822463100059", "AMEX") is False

