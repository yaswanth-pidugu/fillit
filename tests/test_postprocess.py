from app.services.ocr_engine import OCRTextBlock
from app.services.postprocess import (
    extract_card_number_candidates,
    extract_cardholder_name,
    extract_expiry_date,
    parse_ocr_blocks,
    select_best_extraction,
    ExtractedCardData,
)


def test_extract_card_number_candidates_prefers_valid_luhn() -> None:
    blocks = [
        OCRTextBlock(text="4111 1111 1111 1111", confidence=0.95, bbox=[]),
        OCRTextBlock(text="4111 1111 1111 1112", confidence=0.99, bbox=[]),
    ]

    candidates = extract_card_number_candidates(blocks)
    assert candidates[0].card_number == "4111111111111111"


def test_extract_expiry_date_normalizes() -> None:
    blocks = [OCRTextBlock(text="VALID THRU 12/2029", confidence=0.8, bbox=[])]
    assert extract_expiry_date(blocks) == "12/29"


def test_extract_expiry_date_rejects_three_digit_year() -> None:
    blocks = [OCRTextBlock(text="VALID THRU 06/703", confidence=0.8, bbox=[])]
    assert extract_expiry_date(blocks) == "Unknown"


def test_extract_expiry_date_rejects_implausible_future_year() -> None:
    blocks = [OCRTextBlock(text="VALID THRU 08/62", confidence=0.8, bbox=[])]
    assert extract_expiry_date(blocks) == "Unknown"


def test_extract_cardholder_name_filters_noise() -> None:
    blocks = [
        OCRTextBlock(text="VISA PLATINUM", confidence=0.9, bbox=[]),
        OCRTextBlock(text="JOHN DOE", confidence=0.85, bbox=[]),
    ]
    assert extract_cardholder_name(blocks, ["VISA"]) == "JOHN DOE"


def test_parse_ocr_blocks_rejects_non_luhn_pan() -> None:
    blocks = [
        OCRTextBlock(text="4111 1111 1111 1112", confidence=0.99, bbox=[]),
        OCRTextBlock(text="VALID THRU 12/29", confidence=0.99, bbox=[]),
    ]
    parsed = parse_ocr_blocks(blocks)
    assert parsed.card_number == "Unknown"
    assert parsed.network_type == "Unknown"
    assert parsed.bank_name == "Unknown Bank"


def test_select_best_extraction_prefers_more_fields() -> None:
    unknown = ExtractedCardData(
        card_number="Unknown",
        expiry_date="Unknown",
        cardholder_name="Unknown",
        network_type="Unknown",
        bank_name="Unknown Bank",
        confidence=0.0,
    )
    partial = ExtractedCardData(
        card_number="Unknown",
        expiry_date="12/29",
        cardholder_name="JOHN DOE",
        network_type="Unknown",
        bank_name="Unknown Bank",
        confidence=0.6,
    )

    selected_variant, selected, _ = select_best_extraction(
        [
            ("threshold", unknown, []),
            ("raw", partial, [OCRTextBlock(text="JOHN DOE", confidence=0.9, bbox=[])]),
        ]
    )
    assert selected_variant == "raw"
    assert selected.expiry_date == "12/29"


