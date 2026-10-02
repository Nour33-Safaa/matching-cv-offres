import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.preprocessing import basic_clean, clean_text


def test_basic_clean_removes_email():
    text = "Contact me at john.doe@email.com"
    result = basic_clean(text)
    assert "@" not in result


def test_basic_clean_removes_phone():
    text = "Call me at +33 6 12 34 56 78"
    result = basic_clean(text)
    assert "12 34" not in result


def test_clean_text_returns_lowercase():
    result = clean_text("Python Developer")
    assert result == result.lower()


def test_clean_text_empty_input():
    assert clean_text("") == ""
    assert clean_text(None) == ""
