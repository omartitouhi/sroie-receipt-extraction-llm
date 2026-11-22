from src.metrics import (classify, field_equal, mcnemar_exact, norm_date, norm_total,
                         parse_json_output, is_supported, token_f1)

SRC = """OJC MARKETING SDN BHD
NO 2 & 4  JALAN BAYU 4
BANDAR SERI ALAM  81750 MASAI
DATE  : 15/01/2019 11:05:16 AM
TOTAL EXCLUDE GST:  180.00
TOTAL:  193.00"""


def test_norms():
    assert norm_total("RM 1,234.50") == 1234.5
    assert norm_date("15/01/2019") == norm_date("15-01-2019") == (15, 1, 2019)
    assert norm_date("15 JAN 2019") == (15, 1, 2019)
    assert norm_date("2019-01-15") == (15, 1, 2019)


def test_field_equal():
    assert field_equal("total", "RM193.00", "193.00")
    assert field_equal("date", "15-01-19", "15/01/2019")
    assert field_equal("company", "ojc marketing sdn. bhd.", "OJC MARKETING SDN BHD")
    assert not field_equal("company", "", "")


def test_parse():
    assert parse_json_output('Sure! {"total": "1.00",}')[1]
    assert parse_json_output("```json\n{\"a\": 1}\n```") == ({"a": 1}, True)
    assert parse_json_output("no json") == (None, False)


def test_supported_and_classify():
    assert is_supported("193.00", SRC, "total")
    assert is_supported("NO 2 & 4, JALAN BAYU 4, BANDAR SERI ALAM", SRC, "address")
    assert not is_supported("999.99", SRC, "total")
    assert classify("total", "180.00", "193.00", SRC, True) == "mauvais_span"
    assert classify("total", "999.99", "193.00", SRC, True) == "fabrication"
    assert classify("total", "999.99", "193.00", SRC, True, {"999 99"}) == "memorisation"
    assert classify("total", None, "193.00", SRC, True) == "omission"
    assert classify("total", "193", "193.00", SRC, False) == "json_invalide"
    assert classify("address", "NO 2 & 4 JALAN BAYU 4", "NO 2 & 4, JALAN BAYU 4, MASAI",
                    SRC, True) == "partiel"


def test_stats():
    assert token_f1("a b c", "a b d") == 2 / 3
    assert mcnemar_exact([1, 1, 1, 0], [1, 1, 1, 0])["p_value"] == 1.0
    assert mcnemar_exact([0] * 10, [1] * 10)["p_value"] < 0.01
