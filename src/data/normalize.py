"""Unicode normalization for Akkadian transliterations."""
import re
import unicodedata

# ASCII to Unicode diacritic mappings
_ASCII_REPLACEMENTS = [
    ("sz", "š"),
    ("SZ", "Š"),
    ("s,", "ṣ"),
    ("S,", "Ṣ"),
    ("t,", "ṭ"),
    ("T,", "Ṭ"),
]

# Subscript digit mapping (2-9 only; 1 is default reading, never subscripted)
_SUBSCRIPT_DIGITS = str.maketrans("23456789", "₂₃₄₅₆₇₈₉")

# Pattern: a syllable (lowercase letters) immediately followed by a digit 2-9
# Must NOT match standalone numbers like "10" or "0.33333"
_SUBSCRIPT_PATTERN = re.compile(r"(?<=[a-zšṣṭḫāēīū])([2-9])(?=\b|[-\s.,;:])")

# Pattern: uppercase determinative braces like {D} or {KI}
_DETERMINATIVE_PATTERN = re.compile(r"\{([A-Z]+)\}")


def _replace_ascii_diacritics(text: str) -> str:
    for ascii_form, unicode_form in _ASCII_REPLACEMENTS:
        text = text.replace(ascii_form, unicode_form)
    return text


def _subscript_digits(text: str) -> str:
    def _sub(match: re.Match) -> str:
        return match.group(1).translate(_SUBSCRIPT_DIGITS)
    return _SUBSCRIPT_PATTERN.sub(_sub, text)


def _normalize_determinatives(text: str) -> str:
    def _det_lower(match: re.Match) -> str:
        content = match.group(1).lower()
        return "{" + content + "}"
    return _DETERMINATIVE_PATTERN.sub(_det_lower, text)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def normalize_transliteration(text: str) -> str:
    """Normalize an Akkadian transliteration to Unicode with diacritics.

    Applies in order:
    1. ASCII to Unicode diacritics (sz->š, s,->ṣ, t,->ṭ)
    2. Subscript digits on syllables (du3->du₃, but not standalone numbers)
    3. Determinative braces to lowercase ({D}->{d})
    4. NFC Unicode normalization
    5. Whitespace normalization
    """
    text = _replace_ascii_diacritics(text)
    text = _subscript_digits(text)
    text = _normalize_determinatives(text)
    text = unicodedata.normalize("NFC", text)
    text = _normalize_whitespace(text)
    return text
