from app.services.bin_detector import BinDetector


def test_detect_network_strict_ranges() -> None:
    detector = BinDetector()
    assert detector.detect_network("4111111111111111") == "VISA"
    assert detector.detect_network("5555555555554444") == "MASTERCARD"
    assert detector.detect_network("2223000048400011") == "MASTERCARD"
    assert detector.detect_network("378282246310005") == "AMEX"
    assert detector.detect_network("6011111111111117") == "DISCOVER"


def test_detect_bank_from_bin() -> None:
    detector = BinDetector({"411111": ("Test Bank", "VISA")})
    assert detector.detect_bank("4111111111111111") == "Test Bank"
    assert detector.detect_bank("5200828282828210") == "Unknown Bank"


