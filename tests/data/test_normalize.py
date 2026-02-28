import pytest
from src.data.normalize import normalize_transliteration


class TestAsciiToDiacritics:
    def test_sz_to_shin(self):
        assert normalize_transliteration("sza-ru-um") == "ša-ru-um"

    def test_s_comma_to_sade(self):
        assert normalize_transliteration("s,a-lim") == "ṣa-lim"

    def test_t_comma_to_tet(self):
        assert normalize_transliteration("t,up-pu") == "ṭup-pu"

    def test_preserves_existing_diacritics(self):
        assert normalize_transliteration("ša-ru-um") == "ša-ru-um"


class TestSubscriptDigits:
    def test_ascii_digit_to_subscript(self):
        assert normalize_transliteration("du3") == "du₃"

    def test_preserves_existing_subscript(self):
        assert normalize_transliteration("du₃") == "du₃"

    def test_bi4_to_subscript(self):
        assert normalize_transliteration("bi4-ma") == "bi₄-ma"

    def test_no_subscript_for_standalone_numbers(self):
        assert normalize_transliteration("10 ma-na") == "10 ma-na"

    def test_no_subscript_for_sign_index_1(self):
        assert normalize_transliteration("du1") == "du1"


class TestSumerograms:
    def test_determinatives_lowercase_braced(self):
        assert normalize_transliteration("{D}UTU") == "{d}UTU"

    def test_preserves_lowercase_determinatives(self):
        assert normalize_transliteration("{d}UTU") == "{d}UTU"

    def test_logograms_stay_uppercase(self):
        assert normalize_transliteration("LUGAL") == "LUGAL"


class TestUnicodeNormalization:
    def test_nfc_normalization(self):
        decomposed = "s\u030c"
        result = normalize_transliteration(decomposed)
        assert result == "š"

    def test_whitespace_normalization(self):
        assert normalize_transliteration("a-na  bi4-ma") == "a-na bi₄-ma"


class TestRealWorldExamples:
    def test_competition_sample(self):
        text = "KIŠIB ma-nu-ba-lúm-a-šur DUMU ṣí-lá-(d)IM"
        assert normalize_transliteration(text) == text

    def test_mixed_conventions(self):
        text = "sza-ru-um LUGAL du3 {D}UTU"
        expected = "ša-ru-um LUGAL du₃ {d}UTU"
        assert normalize_transliteration(text) == expected
