from __future__ import annotations

from pyrate.reference import parse_fu15_summary, relative_errors


def test_parse_polyrate_fu15_summary() -> None:
    table = parse_fu15_summary(
        """
 Summary of forward rate constants (cm**3/molecule-1 s-1) :

    T(K)     TST       CVT      CVT/ZCT   CVT/SCT

   200.00  1.54E-20  1.54E-20  5.57E-20  3.05E-19
   300.00  2.84E-17  2.84E-17  4.66E-17  1.02E-16
   600.00  4.66E-14  4.66E-14  5.22E-14  6.39E-14
"""
    )
    assert table.direction == "forward"
    assert table.columns == ("TST", "CVT", "CVT/ZCT", "CVT/SCT")
    assert table.column("CVT/ZCT") == (5.57e-20, 4.66e-17, 5.22e-14)
    assert max(relative_errors(table.column("TST"), table.column("CVT"))) == 0.0
