"""Canned test of the measurement logic - hand-made probabilities, no model.

Every number below is worked out by hand, so a pass means the code does the arithmetic we agreed on.
Runnable two ways:
    .venv/bin/python test_outcome_leak.py       (plain asserts + summary)
    pytest test_outcome_leak.py                 (test_* discovery)
"""

import math

import torch

from outcome_leak import expected_blame, fraction_removed, fraction_restored, gate, read_curve, softcap


def one_hot(d, p=1.0):
    probs = [0.0] * 10
    probs[d] = p
    return probs


def close(a, b):
    return math.isclose(a, b, abs_tol=1e-9)


def test_expected_blame():
    # all probability on "7" -> blame 7, every bit of it on digits
    assert expected_blame(one_hot(7)) == (7.0, 1.0)
    # the two ends of the scale
    assert expected_blame(one_hot(0)) == (0.0, 1.0)
    assert expected_blame(one_hot(9)) == (9.0, 1.0)
    # 50/50 between "2" and "4" -> 3
    probs = [0.0] * 10
    probs[2], probs[4] = 0.5, 0.5
    assert expected_blame(probs) == (3.0, 1.0)
    # only half the mass on digits (the rest on, say, a word): blame still 3, mass 0.5
    probs[2], probs[4] = 0.25, 0.25
    assert expected_blame(probs) == (3.0, 0.5)
    # equal on every digit -> the middle, 4.5
    blame, mass = expected_blame([0.1] * 10)
    assert close(blame, 4.5) and close(mass, 1.0)
    # nothing on any digit -> no blame number at all, not a fake 0
    blame, mass = expected_blame([0.0] * 10)
    assert math.isnan(blame) and mass == 0.0
    # the wrong number of digits is a bug, not a result
    try:
        expected_blame([0.5, 0.5])
    except AssertionError:
        pass
    else:
        raise AssertionError("expected_blame accepted 2 probabilities")


def test_fraction_restored():
    # the worked example: no harm 3.0, harm 5.0, patched 4.5 -> 0.75 of the leak back
    assert close(fraction_restored(4.5, 3.0, 5.0), 0.75)
    assert close(fraction_restored(3.0, 3.0, 5.0), 0.0)  # swap did nothing
    assert close(fraction_restored(5.0, 3.0, 5.0), 1.0)  # swap brought all of it
    # a negative leak (less blame after harm) still reads the same way
    assert close(fraction_restored(4.0, 5.0, 3.0), 0.5)


def test_fraction_removed():
    # the notes.md example: no harm 3.56, harm 6.28, harm run with no-harm notes drops to 4.00 -> 0.84 removed
    assert close(fraction_removed(4.00, 3.56, 6.28), 2.28 / 2.72)
    assert close(fraction_removed(6.28, 3.56, 6.28), 0.0)  # swap removed nothing
    assert close(fraction_removed(3.56, 3.56, 6.28), 1.0)  # swap removed the whole leak


def test_softcap():
    # the notes.md table, worked by hand: 30 x tanh(score / 30)
    assert softcap(0.0, 30.0) == 0.0
    assert close(softcap(30.0, 30.0), 30 * math.tanh(1))  # 22.85
    assert round(softcap(30.0, 30.0), 2) == 22.85
    assert round(softcap(-30.0, 30.0), 2) == -22.85  # symmetric
    assert round(softcap(3.0, 30.0), 2) == 2.99  # small scores barely move
    assert round(softcap(100.0, 30.0), 2) == 29.92  # big ones never pass 30
    # the same on a tensor - that is what the logit lens feeds it
    out = softcap(torch.tensor([0.0, 30.0, 100.0]), 30.0)
    assert [round(x, 2) for x in out.tolist()] == [0.0, 22.85, 29.92]


def test_gate():
    assert gate((3.0, 0.95), (3.31, 0.95)) == []  # leak 0.31 -> go on
    assert gate((3.31, 0.95), (3.0, 0.95)) == []  # leak -0.31 -> go on too (two-sided)
    assert gate((3.0, 0.95), (3.29, 0.95)) == ["|leak| 0.290 < 0.3"]  # too small -> stop
    assert gate((3.0, 0.89), (5.0, 0.95)) == ["no_harm digit mass 0.890 < 0.9"]  # not a number answer -> stop
    assert gate((3.0, 0.95), (5.0, 0.5)) == ["harm digit mass 0.500 < 0.9"]


def test_read_curve():
    mass = [0.95] * 8
    # sharp: 0.1 at layer 2 -> 0.9 at layer 4 (2 layers apart); entry at 3 (first >= 0.5)
    assert read_curve([0, 0, 0.1, 0.6, 0.9, 1, 1, 1], mass) == (3, "sharp", [])
    # gradual: climbs 0.15 per layer, never under 0.2 -> over 0.8 within 3 layers; entry at 4 (0.6)
    assert read_curve([0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1], mass) == (4, "gradual", [])
    # never reaches 0.5 -> no entry layer
    assert read_curve([0, 0.1, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3], mass) == (None, "gradual", [])
    # layer 3 is flagged (model stopped answering with a number) -> skipped: entry moves to 4
    assert read_curve([0, 0, 0.1, 0.6, 0.9, 1, 1, 1], [0.95, 0.95, 0.95, 0.5, 0.95, 0.95, 0.95, 0.95]) == (
        4,
        "sharp",
        [3],
    )


if __name__ == "__main__":
    tests = [test_expected_blame, test_fraction_restored, test_fraction_removed, test_softcap, test_gate, test_read_curve]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)}/{len(tests)} passed")
