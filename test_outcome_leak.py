"""Canned test of the measurement logic - hand-made probabilities, no model.

Every number below is worked out by hand, so a pass means the code does the arithmetic we agreed on.
Runnable two ways:
    .venv/bin/python test_outcome_leak.py       (plain asserts + summary)
    pytest test_outcome_leak.py                 (test_* discovery)
"""

import math

import torch

from outcome_leak import (
    cosine, expected_blame, fraction_removed, fraction_restored, gate, largest, outcome_direction, read_curve,
    read_removal, remove, riley_gate, shrink, softcap,
)


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


def vec(*xs):
    return torch.tensor(xs, dtype=torch.float64)  # exact enough for the 1e-9 check


def test_outcome_direction():
    # the notes.md toy, notes of 3 numbers [outcome, child-and-car stuff, hospital-or-code stuff]
    sam, alex = vec(1.0, 0.0, 0.6), vec(1.0, 0.0, -0.4)
    u = outcome_direction([sam, alex])
    # average [1.0, 0.0, 0.1], length sqrt(1.01) = 1.00499 -> [0.99504, 0, 0.09950]
    assert torch.allclose(u, vec(1.0, 0.0, 0.1) / math.sqrt(1.01))
    assert close(u.norm().item(), 1.0)  # length 1: only which way
    # Dana's own difference [1.0, 0.8, 0.0]: cosine = 0.99504 / sqrt(1.64) = 0.777 - alike, not the same
    assert round(cosine(u, vec(1.0, 0.8, 0.0)), 3) == 0.777
    assert close(cosine(sam, sam), 1.0) and close(cosine(sam, -sam), -1.0)
    # row 0: the last token is the same in every leg, so the differences are all zeros -> no direction
    assert outcome_direction([vec(0, 0, 0), vec(0, 0, 0)]) is None
    assert math.isnan(cosine(vec(0, 0, 0), sam))


def test_remove():
    # the notes.md table, u = the outcome slider [1, 0, 0]
    u = vec(1.0, 0.0, 0.0)
    assert torch.allclose(remove(vec(2.0, 0.8, 0.3), u), vec(0.0, 0.8, 0.3))  # Dana harm: 2.0 along u -> 0
    assert torch.allclose(remove(vec(1.0, 0.0, 0.3), u), vec(0.0, 0.0, 0.3))  # Dana no harm: 1.0 along u -> 0
    # with the real toy u (not rounded): nothing is left along u, and the rest is untouched
    u = outcome_direction([vec(1.0, 0.0, 0.6), vec(1.0, 0.0, -0.4)])
    h = vec(2.0, 0.8, 0.3)
    after = remove(h, u)
    assert close((after @ u).item(), 0.0)
    assert close(after[1].item(), 0.8)  # the child-and-car number does not point along u
    # h . u = (2.0 + 0.03) / 1.00499 = 2.01993; after = h - 2.01993 x u = [-0.0099, 0.8, 0.0990]
    assert torch.allclose(after, vec(2.0 - 2.03 / 1.01, 0.8, 0.3 - 0.203 / 1.01))
    assert remove(h, None) is h  # no direction (row 0) -> notes unchanged


def test_remove_every_token():
    # the step a toy: 4 tokens, notes [outcome, other, other], u = the outcome slider
    u = vec(1.0, 0.0, 0.0)
    notes = torch.stack([vec(0.0, 0.5, 0.2),  # "Dana"
                         vec(0.1, 0.3, 0.9),  # "phone"
                         vec(3.0, 0.4, 0.1),  # "child-hurt"
                         vec(2.0, 0.8, 0.3)])  # the last token
    # last token only: the last row loses its outcome, "child-hurt" keeps its 3.0
    last = notes.clone()
    last[-1] = remove(notes[-1], u)
    assert torch.allclose(last[-1], vec(0.0, 0.8, 0.3)) and last[2, 0].item() == 3.0
    # a later layer copies 30% of "child-hurt"'s outcome into the last token: 0.3 x 3.0 = 0.9 comes back
    assert close(0.3 * (last[2] @ u).item(), 0.9)
    # every token: every row loses its outcome, so there is nothing left to copy
    every = remove(notes, u)
    assert torch.allclose(every, torch.stack([vec(0.0, 0.5, 0.2), vec(0.0, 0.3, 0.9), vec(0.0, 0.4, 0.1),
                                              vec(0.0, 0.8, 0.3)]))
    assert close(0.3 * (every[2] @ u).item(), 0.0)
    # with the real toy u (not rounded): all four tokens end with 0 along u, each the same as removing it alone
    u = outcome_direction([vec(1.0, 0.0, 0.6), vec(1.0, 0.0, -0.4)])
    every = remove(notes, u)
    assert torch.allclose(every @ u, vec(0, 0, 0, 0))
    assert all(torch.allclose(every[i], remove(notes[i], u)) for i in range(4))
    assert remove(notes, None) is notes  # no direction (row 0) -> notes unchanged


def test_random_control():
    # 10 random directions at one row: the largest move decides, sign kept
    assert largest([0.02, -0.01, 0.03, 0.0, -0.02, 0.01, 0.0, 0.02, -0.03, 0.03]) == 0.03
    assert largest([0.02, -0.12, 0.05]) == -0.12  # one seed moved the leak 12% down
    # fed to the rules: row 0 every seed under 10% -> holds; row 1 one seed at -12% -> control fails there
    moves = [largest([0.02, -0.01, 0.03]), largest([0.02, -0.12, 0.05])]
    assert read_removal([0.6, 0.6], [0.95, 0.95], [0.9, 0.9], [0.95, 0.95], moves) == ([0, 1], [0, 1], [0])


def test_shrink_and_rules():
    # the notes.md "read three things" table: Dana leak 2.72 -> 0.4 = 85% removed
    assert round(shrink(2.72, 0.4), 2) == 0.85
    assert close(shrink(2.72, 2.72), 0.0) and close(shrink(2.72, 0.0), 1.0)
    assert round(shrink(2.72, -0.272), 2) == 1.1  # the leak flipped: more than all of it removed
    # Riley gap 4.0 -> 3.6 keeps 90% (good), -> 0.4 keeps 10% (judgment broken)
    assert round(1 - shrink(4.0, 3.6), 2) == 0.9 and round(1 - shrink(4.0, 0.4), 2) == 0.1
    # gate: Riley careful 2.0, careless 6.0 -> gap 4.0, go on; gap 0.2 -> stop
    assert riley_gate((2.0, 0.95), (6.0, 0.95)) == []
    assert riley_gate((2.0, 0.95), (2.2, 0.95)) == ["Riley gap careless - careful 0.200 < 0.3"]
    assert riley_gate((6.0, 0.95), (2.0, 0.95)) != []  # careless blamed LESS than careful -> no real gap either
    # rows 0-6, worked by hand:
    #   row 0: shrink 0.0                                  -> not taken out
    #   row 1: shrink 0.6, but Dana digit mass 0.5         -> not taken out (not answering with a number)
    #   row 2: shrink 0.6, Riley keeps 0.5                 -> taken out, grounds broken
    #   row 3: shrink 0.6, Riley keeps 0.9, random 0.15    -> taken out + kept, control fails
    #   row 4: shrink 0.85, Riley keeps 0.9, random -0.05  -> all three hold
    #   row 5: shrink 0.5 exactly, keeps 0.8 exactly, both masses 0.9 exactly, random 0.0 -> all three hold (>=)
    #   row 6: shrink 0.85, Riley keeps 0.9, but Riley digit mass 0.5 -> taken out, "kept" unreadable
    shrinks = [0.0, 0.6, 0.6, 0.6, 0.85, 0.5, 0.85]
    masses = [0.95, 0.5, 0.95, 0.95, 0.95, 0.9, 0.95]
    kept = [1.0, 1.0, 0.5, 0.9, 0.9, 0.8, 0.9]
    riley_masses = [0.95, 0.95, 0.95, 0.95, 0.95, 0.9, 0.5]
    moves = [0.0, 0.0, 0.0, 0.15, -0.05, 0.0, 0.0]
    assert read_removal(shrinks, masses, kept, riley_masses, moves) == ([2, 3, 4, 5, 6], [3, 4, 5], [4, 5])


if __name__ == "__main__":
    tests = [test_expected_blame, test_fraction_restored, test_fraction_removed, test_softcap, test_gate, test_read_curve,
             test_outcome_direction, test_remove, test_remove_every_token, test_random_control, test_shrink_and_rules]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)}/{len(tests)} passed")
