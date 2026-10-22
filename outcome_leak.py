"""Dana outcome leak in gemma-2-2b-it: measure it, then find where the outcome reaches the answer.

Plain explanation: notes.md. Agreed plan and stop rules: next-session-prompt.md.
Top half: the measurement logic (tested by test_outcome_leak.py, no model). Bottom half: the model.

    .venv/bin/python outcome_leak.py smoke      step 0 - look at the machinery, three honesty checks
"""

import json
import math
import pathlib
import sys

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


# ------------------------------------------------------------------------------------------------ the model

import torch  # noqa: E402 - below the logic on purpose: the canned test reads the top half first

MODEL = "google/gemma-2-2b-it"
DEVICE = "mps"
FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "moral-luck-02.json"  # verbatim copy, theory-grounds-testing
PREFILL = "BLAME-FOR-CHOICE: "  # gemma's answer starts with this, so the next token is the number
LEGS = ("no_harm", "harm")
SANITY_TOL = 1e-3  # an honesty check passes if expected blame moves by at most this (fixed before the smoke)


def prompts():
    """The two Dana prompts: leg + question + answer format, joined as moral-luck-02 did, 0-10 -> 0-9."""
    fx = json.loads(FIXTURE.read_text())
    case = fx["cases"]["A"]
    contract = fx["output_contract"]
    assert contract.count(" to 10") == 2, "the 0-10 scale should appear exactly twice"
    contract = contract.replace(" to 10", " to 9")
    question = fx["question"].format(name=case["name"])
    return {leg: "\n\n".join([case["legs"][leg], question, contract.format(name=case["name"])]) for leg in LEGS}


def load():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    # eager attention: gemma 2 caps its attention scores, and only the eager path does that.
    # device_map loads straight onto the GPU - loading to memory first and then copying needs ~2x on 8 GB.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="eager", device_map=DEVICE
    )
    return tok, model.eval()


def encode(tok, user):
    """Chat template + the prefill, as token ids. The template already starts with <bos>."""
    text = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
    return tok(text + PREFILL, add_special_tokens=False, return_tensors="pt").input_ids.to(DEVICE)


def digit_ids(tok):
    ids = [tok.convert_tokens_to_ids(str(d)) for d in DIGITS]
    assert [tok.convert_ids_to_tokens(i) for i in ids] == [str(d) for d in DIGITS], "a digit is not one token"
    return ids


def run(model, ids, patch=None):
    """One pass. Returns the last token's notes at every layer and the next-token probabilities.

    Notes row 0 = before layer 1, row L = after layer L (27 rows of 2304 numbers).
    patch = (L, notes) overwrites the last token's notes at row L before the model reads on.
    """
    layers = model.model.layers
    saved = [None] * (len(layers) + 1)

    def at(L):
        def touch(hs):
            if patch is not None and patch[0] == L:
                hs[0, -1] = patch[1].to(hs.device, hs.dtype)
            saved[L] = hs[0, -1].float().cpu()

        return touch

    # row 0 is the input of layer 1; rows 1-26 are each layer's output. Hooks edit in place, return nothing.
    hooks = [layers[0].register_forward_pre_hook(lambda m, a, k, t=at(0): t(a[0] if a else k["hidden_states"]),
                                                 with_kwargs=True)]
    hooks += [layer.register_forward_hook(lambda m, a, o, t=at(L + 1): t(o[0] if isinstance(o, tuple) else o))
              for L, layer in enumerate(layers)]
    try:
        with torch.no_grad():
            logits = model(ids, logits_to_keep=1).logits[0, -1]
    finally:
        for h in hooks:
            h.remove()
    return torch.stack(saved), torch.softmax(logits.float(), -1).cpu()


def smoke():
    import transformers

    print(f"torch {torch.__version__} | transformers {transformers.__version__} | mps {torch.backends.mps.is_available()}")
    tok, model = load()
    print(f"model {MODEL} revision {getattr(model.config, '_commit_hash', None)} | dtype {model.dtype} | "
          f"device {model.device} | layers {len(model.model.layers)} | notes size {model.config.hidden_size} | "
          f"mps memory {torch.mps.current_allocated_memory() / 1e9:.2f} GB")
    dids = digit_ids(tok)
    print("digit token ids:", dict(zip(DIGITS, dids)))
    ids = {leg: encode(tok, user) for leg, user in prompts().items()}

    def blame(probs):
        return expected_blame([probs[i].item() for i in dids])

    print("\n== 1. the prompts as gemma sees them (position, token id, token)")
    for leg in LEGS:
        print(f"\n-- {leg}: {ids[leg].shape[1]} tokens")
        for i, t in enumerate(ids[leg][0].tolist()):
            print(f"{i:4d} {t:7d} {tok.convert_ids_to_tokens(t)!r}")
    a, b = ids["no_harm"][0].tolist(), ids["harm"][0].tolist()
    start = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
    end = next(i for i, (x, y) in enumerate(zip(a[::-1], b[::-1])) if x != y)
    print(f"\nshared start {start} tokens | shared end {end} tokens | "
          f"differing middle: no_harm {len(a) - start - end}, harm {len(b) - start - end}")
    assert a[-1] == b[-1] and tok.convert_ids_to_tokens(a[-1]) == "▁", "the read position must be the space token"

    print("\n== 2. what gemma writes after the prefill (greedy, 40 tokens)")
    for leg in LEGS:
        with torch.no_grad():
            out = model.generate(ids[leg], max_new_tokens=40, do_sample=False)
        print(f"\n-- {leg}:\n{PREFILL}{tok.decode(out[0, ids[leg].shape[1]:])}")

    print("\n== 3. next-token probabilities at the read position")
    base = {leg: run(model, ids[leg]) for leg in LEGS}
    for leg in LEGS:
        probs = base[leg][1]
        top = torch.topk(probs, 10)
        print(f"\n-- {leg}: top 10")
        for p, i in zip(top.values.tolist(), top.indices.tolist()):
            print(f"  {p:.4f} {tok.convert_ids_to_tokens(i)!r}")
        print("  digits: " + " ".join(f"{d}:{probs[i].item():.4f}" for d, i in zip(DIGITS, dids)))
        b_, m_ = blame(probs)
        print(f"  expected blame {b_:.4f} | digit mass {m_:.4f}")

    print("\n== 4. the notes at the last token (row 0 = before layer 1, row L = after layer L)")
    for leg in LEGS:
        notes = base[leg][0]
        print(f"-- {leg}: shape {tuple(notes.shape)}")
        for L in (0, 1, 13, 26):
            print(f"  row {L:2d}: first 5 numbers {[round(x, 3) for x in notes[L, :5].tolist()]} | "
                  f"size {notes[L].norm():.1f}")
    gap = (base["harm"][0] - base["no_harm"][0]).norm(dim=1)
    size = base["no_harm"][0].norm(dim=1)
    print("how far apart the two legs' notes are (0 = identical; relative = gap / size of no_harm's notes):")
    for L in range(len(gap)):
        print(f"  row {L:2d}: gap {gap[L]:8.2f} | relative {gap[L] / size[L]:.3f}")

    print("\n== 5. honesty checks for the swap")
    no_harm, harm = blame(base["no_harm"][1])[0], blame(base["harm"][1])[0]
    own = max(abs(blame(run(model, ids["no_harm"], (L, base["no_harm"][0][L]))[1])[0] - no_harm) for L in range(27))
    last = blame(run(model, ids["no_harm"], (26, base["harm"][0][26]))[1])[0]
    first = blame(run(model, ids["no_harm"], (0, base["harm"][0][0]))[1])[0]
    checks = [
        ("a) no_harm swapped with its OWN notes, every row 0-26: largest change", own),
        (f"b) harm notes at row 26 into no_harm: {last:.4f} vs harm {harm:.4f}, difference", abs(last - harm)),
        (f"c) harm notes at row 0 into no_harm: {first:.4f} vs no_harm {no_harm:.4f}, difference", abs(first - no_harm)),
    ]
    for what, d in checks:
        print(f"{what} {d:.6f} -> {'PASS' if d <= SANITY_TOL else 'FAIL'} (tolerance {SANITY_TOL})")
    passed = sum(d <= SANITY_TOL for _, d in checks)
    print(f"\nSMOKE honesty checks: {passed}/{len(checks)} pass")
    return passed == len(checks)


if __name__ == "__main__":
    sys.exit(0 if {"smoke": smoke}[sys.argv[1]]() else 1)
