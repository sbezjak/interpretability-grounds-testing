"""Dana outcome leak in gemma-2-2b-it: measure it, then find where the outcome reaches the answer.

Plain explanation: notes.md. Agreed plan and stop rules: next-session-prompt.md.
This file starts with the measurement logic only (tested by test_dana_patching.py, no model);
the model part is added after the canned test passes.
"""

import math

DIGITS = range(10)  # the 0-9 blame scale; every digit is one gemma token
LEAK_MIN = 0.3  # |leak| under this = no leak -> stop (fixed 2026-10-22, before any run)
MASS_MIN = 0.9  # digit mass under this = not answering with a number -> stop / flag
ENTRY = 0.5  # entry layer = first layer whose fraction restored reaches this
SHARP_LOW, SHARP_HIGH, SHARP_WITHIN = 0.2, 0.8, 3  # sharp = under 0.2 -> over 0.8 within 3 layers


def expected_blame(digit_probs):
    """digit_probs[d] = the model's probability that the next token is digit d.

    Returns (expected blame, digit mass). The blame is renormalised over the 10 digits,
    so probability on other tokens neither pulls it up nor down - it shows up in the mass.
    """
    assert len(digit_probs) == 10, f"need 10 digit probabilities, got {len(digit_probs)}"
    mass = sum(digit_probs)
    if mass == 0:
        return math.nan, 0.0
    return sum(d * p for d, p in zip(DIGITS, digit_probs)) / mass, mass


def fraction_restored(patched, no_harm, harm):
    """How much of the leak one swap brought back: 0 = none, 1 = all of it."""
    return (patched - no_harm) / (harm - no_harm)


def gate(no_harm, harm):
    """Step 1 stop rules. Each leg is (expected blame, digit mass). Returns the stop reasons; empty = go on."""
    reasons = []
    for name, (_, mass) in (("no_harm", no_harm), ("harm", harm)):
        if mass < MASS_MIN:
            reasons.append(f"{name} digit mass {mass:.3f} < {MASS_MIN}")
    leak = harm[0] - no_harm[0]
    if abs(leak) < LEAK_MIN:
        reasons.append(f"|leak| {abs(leak):.3f} < {LEAK_MIN}")
    return reasons


def read_curve(fractions, masses):
    """Apply the reading rules to one patching curve (index = layer, 0 = before any layer).

    Returns (entry layer or None, "sharp" or "gradual", flagged layers). A flagged layer
    (digit mass under MASS_MIN) does not count for the entry layer or for sharpness.
    """
    flagged = [i for i, m in enumerate(masses) if m < MASS_MIN]
    ok = [i for i in range(len(fractions)) if i not in flagged]
    entry = next((i for i in ok if fractions[i] >= ENTRY), None)
    sharp = any(
        fractions[i] < SHARP_LOW and fractions[j] > SHARP_HIGH
        for i in ok
        for j in ok
        if i < j <= i + SHARP_WITHIN
    )
    return entry, "sharp" if sharp else "gradual", flagged
