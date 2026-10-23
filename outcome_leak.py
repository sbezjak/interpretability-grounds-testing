"""Dana outcome leak in gemma-2-2b-it: measure it, then find where the outcome reaches the answer.

Plain explanation: notes.md. Agreed plan and stop rules: next-session-prompt.md.
Top half: the measurement logic (tested by test_outcome_leak.py, no model). Bottom half: the model.

    .venv/bin/python outcome_leak.py smoke              step 0 - look at the machinery, three honesty checks
    .venv/bin/python outcome_leak.py patching <name>    step 1 (stop rules) -> logit lens -> step 2, both directions
    .venv/bin/python outcome_leak.py lens-from-log <name>   redraw the lens picture from that run's log, no model
    .venv/bin/python outcome_leak.py removal <name>     step 3 - take the outcome direction out, Riley and random checks
    .venv/bin/python outcome_leak.py removal-from-log <name>   redraw the step 3 picture from that run's log, no model
    .venv/bin/python outcome_leak.py removal-sturdy <name>   step a - last token vs every token, 10 random directions
    .venv/bin/python outcome_leak.py removed-smoke      step c smoke - the three reads on Kai, fixed gemma vs step a
    .venv/bin/python outcome_leak.py removed-cases <name>   step c - fixed gemma on new cases E-G, Dana, Riley
    .venv/bin/python outcome_leak.py normal-test <name>   the paid models' moral-luck test on plain and fixed gemma
"""

import contextlib
import faulthandler
import json
import math
import pathlib
import signal
import statistics
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


def fraction_removed(patched, no_harm, harm):
    """The reverse swap (no-harm notes into the harm run): 0 = blame did not fall, 1 = the whole leak gone."""
    return fraction_restored(patched, harm, no_harm)


def softcap(score, cap):
    """Gemma's squash: 30 x tanh(score / 30), so no score passes the cap. Works on a number or a tensor."""
    return cap * (math.tanh(score / cap) if isinstance(score, (int, float)) else (score / cap).tanh())


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


# step 3, take the leak out (agreed 2026-10-23, before the run)
GAP_MIN = 0.3  # Riley careless - careful under this at baseline = grounds check unreadable -> stop
SHRINK_MIN = 0.5  # taken out = Dana's leak shrinks by at least this
KEEP_MIN = 0.8  # grounds kept = Riley's gap keeps at least this, at the same row
CONTROL_MAX = 0.1  # a random direction must move Dana's leak by less than this


def unit(v):
    """v scaled to length 1 - which way, not how much. None if v is all zeros: there is no way to point."""
    n = v.norm()
    return None if n == 0 else v / n


def outcome_direction(diffs):
    """The average of the build cases' (harm - no_harm) notes, length 1 - the outcome direction u."""
    return unit(sum(diffs) / len(diffs))


def remove(h, u):
    """h without its part along u: h - (h . u) u. u = None (no direction) leaves h as it is.

    h is one token's notes, or a table with one row per token - then every token loses its part along u.
    """
    return h if u is None else h - (h @ u)[..., None] * u


def largest(moves):
    """The move furthest from 0, sign kept - the random control passes only if even this one is small."""
    return max(moves, key=abs)


def cosine(a, b):
    """How alike two directions are: 1 = the same way, 0 = unrelated, -1 = opposite. nan if either is all zeros."""
    ua, ub = unit(a), unit(b)
    return math.nan if ua is None or ub is None else (ua @ ub).item()


def shrink(before, after):
    """How much of a leak (or gap) the removal took away: 0 = none, 1 = all of it, over 1 = it flipped sign."""
    return 1 - after / before


def riley_gate(careful, careless):
    """Step 3 gate. Each leg is (expected blame, digit mass). Returns the stop reasons; empty = go on."""
    gap = careless[0] - careful[0]
    return [f"Riley gap careless - careful {gap:.3f} < {GAP_MIN}"] if gap < GAP_MIN else []


def read_removal(shrinks, masses, kept, riley_masses, moves):
    """Lines 3-5 per row: Dana's leak shrinks >= 0.5 with both legs' digit mass >= 0.9 (taken out), Riley's gap
    keeps >= 0.8 with both legs' digit mass >= 0.9 (grounds kept), the random direction moves Dana's leak by < 0.1
    (control). masses / riley_masses = the lower of that case's two legs.

    Returns (rows taken out, rows taken out + grounds kept, rows where all three hold).
    """
    out = [L for L, (s, m) in enumerate(zip(shrinks, masses)) if s >= SHRINK_MIN and m >= MASS_MIN]
    grounds = [L for L in out if kept[L] >= KEEP_MIN and riley_masses[L] >= MASS_MIN]
    return out, grounds, [L for L in grounds if abs(moves[L]) < CONTROL_MAX]


# step c, the removed model on new cases (agreed 2026-10-23, before the smoke)
LIABILITY = (" OWES-COMPENSATION", " OWES-NOTHING")  # the two answers the output contract allows, after "LIABILITY:"


def phrase_prob(token_probs):
    """The probability of a whole phrase = its tokens' probabilities, each given the ones before, multiplied."""
    return math.prod(token_probs)


def liability_read(p_owes, p_nothing):
    """Returns (P owes, mass): P owes renormalised over the two allowed answers (1 = sure it owes, 0 = sure it
    doesn't), mass = how much of gemma's probability the two answers hold together - like digit mass."""
    mass = p_owes + p_nothing
    return (math.nan if mass == 0 else p_owes / mass), mass


LEFT_MAX = 0.5  # leak out = at most half of the plain leak is left, by size - a flipped leak counts by its size
# (line 4 clarified to this by Sara after the smoke showed Kai's leak flip sign, before the run)


def left(before, after):
    """How much of a leak is left, by size: 0 = none, 1 = all of it, over 1 = bigger. -0.35 after +0.55 = 0.64."""
    return abs(after) / abs(before)


def case_gate(gaps, masses):
    """Plain gemma, one case. gaps = (choice leak, harm-blame gap, liability gap), each harm - no_harm; masses = every
    read's mass. Returns the reasons to drop the case; empty = it can be tested."""
    choice, harm, liability = gaps
    reasons = [f"|choice leak| {abs(choice):.3f} < {LEAK_MIN}"] if abs(choice) < LEAK_MIN else []
    reasons += [f"harm-blame gap {harm:.3f} < {GAP_MIN}"] if harm < GAP_MIN else []
    reasons += [f"liability gap {liability:.3f} < {GAP_MIN}"] if liability < GAP_MIN else []
    reasons += [f"a read's mass {min(masses):.3f} < {MASS_MIN}"] if min(masses) < MASS_MIN else []
    return reasons


def read_case(plain, fixed, fixed_masses, moves):
    """Line 4 for one case. plain / fixed = (choice leak, harm-blame gap, liability gap); fixed_masses = fixed gemma's
    (choice masses, harm + liability masses); moves = what each random direction did to the choice leak.

    Returns (left, harm kept, liability kept, largest random move, leak out, outcome kept).
    """
    gone = left(plain[0], fixed[0])
    harm_kept, liability_kept = fixed[1] / plain[1], fixed[2] / plain[2]
    move = largest(moves)
    out = gone <= LEFT_MAX and abs(move) < CONTROL_MAX and min(fixed_masses[0]) >= MASS_MIN
    kept = harm_kept >= KEEP_MIN and liability_kept >= KEEP_MIN and min(fixed_masses[1]) >= MASS_MIN
    return gone, harm_kept, liability_kept, move, out, kept


# the normal test: the paid models' moral-luck test on plain and fixed gemma (agreed 2026-10-23, before the run)
MIN_READABLE = 8  # a story needs >= 8 of its 10 answers with a readable choice number, or the case is not scored


def readable(answers, field):
    """How many sampled answers carry a readable number in field (parsed by the paid runs' own scorer)."""
    return sum(a.get(field) is not None for a in answers)


def fixed_case(plain, fixed, riley_moves):
    """Line 5: a case is fixed if plain gemma bleeds, fixed gemma does not (and answered), and Riley still moves."""
    return plain == "blame_bleed" and fixed not in ("blame_bleed", "non_compliant") and riley_moves


# ------------------------------------------------------------------------------------------------ the model

import torch  # noqa: E402 - below the logic on purpose: the canned test reads the top half first

MODEL = "google/gemma-2-2b-it"
DEVICE = "mps"
FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "moral-luck-02.json"  # verbatim copy, theory-grounds-testing
FIXTURE_06 = FIXTURE.parent / "moral-luck-06-sep.json"  # verbatim copy, judge-outcome-bias-probe d4b5579: cases E-G
PREFILL = "BLAME-FOR-CHOICE: "  # gemma's answer starts with this, so the next token is the number
LEGS = ("no_harm", "harm")
SANITY_TOL = 1e-3  # an honesty check passes if expected blame moves by at most this (fixed before the smoke)


def prompts(key="A", fixture=FIXTURE):
    """One case's prompts (A = Dana): leg + question + answer format, joined as moral-luck-02 did, 0-10 -> 0-9."""
    fx = json.loads(fixture.read_text())
    case = fx["cases"][key]
    contract = fx["output_contract"]
    assert contract.count(" to 10") == 2, "the 0-10 scale should appear exactly twice"
    contract = contract.replace(" to 10", " to 9")
    question = fx["question"].format(name=case["name"])
    return {leg: "\n\n".join([text, question, contract.format(name=case["name"])]) for leg, text in case["legs"].items()}


def load():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    # eager attention: gemma 2 caps its attention scores, and only the eager path does that.
    # device_map loads straight onto the GPU - loading to memory first and then copying needs ~2x on 8 GB.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="eager", device_map=DEVICE
    )
    return tok, model.eval()


def encode(tok, user, prefill=PREFILL):
    """Chat template + the prefill (the start of gemma's answer), as token ids. The template already starts with <bos>."""
    text = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
    return tok(text + prefill, add_special_tokens=False, return_tensors="pt").input_ids.to(DEVICE)


def digit_ids(tok):
    ids = [tok.convert_tokens_to_ids(str(d)) for d in DIGITS]
    assert [tok.convert_ids_to_tokens(i) for i in ids] == [str(d) for d in DIGITS], "a digit is not one token"
    return ids


def run(model, ids, patch=None):
    """One pass. Returns the last token's notes at every layer and the next-token probabilities.

    Notes row 0 = before layer 1, row L = after layer L (27 rows of 2304 numbers).
    patch = (L, notes) overwrites the last token's notes at row L before the model reads on.
    patch = (L, edit) with edit a function: every token's notes at row L become edit(notes), worked in float.
    """
    layers = model.model.layers
    saved = [None] * (len(layers) + 1)

    def at(L):
        def touch(hs):
            if patch is not None and patch[0] == L and callable(patch[1]):
                hs[0] = patch[1](hs[0].float()).to(hs.dtype)
            elif patch is not None and patch[0] == L:
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


def lens(model, notes):
    """Logit lens: gemma's final reader (norm -> scores -> squash) on every row of notes, as gemma's own code does it."""
    with torch.no_grad():
        h = model.model.norm(notes.to(DEVICE, model.dtype))
        scores = softcap(model.lm_head(h), model.config.final_logit_softcapping)
    return torch.softmax(scores.float(), -1).cpu()


COLORS = {"no_harm": "#2a78d6", "harm": "#eb6834"}  # blue / orange, colorblind-checked; a color = whose notes
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def pictures(pngs, digits, view, curves):
    """The PNGs named in pngs. Plain data in, so they can be drawn from made-up numbers or from a log.

    digits[leg] = 10 probabilities, view[leg] = 27 (blame, digit mass) from the lens, curves[direction] = 27 fractions.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": MUTED, "axes.edgecolor": GRID,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                         "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False})
    label = {"no_harm": "no harm", "harm": "harm"}

    def save(fig, ax, key, title, sub):
        ax.set_title(f"{title}\n", loc="left", fontsize=12, fontweight="bold")
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=9, color=MUTED)
        fig.tight_layout()
        fig.savefig(pngs[key], dpi=150, facecolor="white")
        plt.close(fig)

    # 1. what gemma thinks of each blame number, both legs side by side
    if "digits" in pngs:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        w = 0.38
        for k, leg in enumerate(LEGS):
            blame, _ = expected_blame(digits[leg])
            ax.bar([d + (k - 0.5) * (w + 0.02) for d in DIGITS], digits[leg], w, color=COLORS[leg],
                   label=f"{label[leg]} - expected blame {blame:.2f}")
        ax.set(xticks=list(DIGITS), xlabel="blame number gemma could write next (0 = no blame at all)",
               ylabel="probability", ylim=(0, 1))
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper left")
        save(fig, ax, "digits", "Gemma's answer, before it writes it",
             f"probability of each blame number for Dana's choice - {MODEL}")

    rows = range(len(view["no_harm"] if view else curves["forward"]))
    if "lens" in pngs:
        # 2. logit lens: the answer read after every layer - only where the reader actually sees a number
        readable = [L for L in rows if min(view[leg][L][1] for leg in LEGS) >= MASS_MIN]
        fig, (ax, low) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        if readable and readable[0] > 0:
            for a in (ax, low):
                a.axvspan(-0.5, readable[0] - 0.5, color=GRID, alpha=0.5, linewidth=0)
            ax.text((readable[0] - 1) / 2, 4.5, f"rows 0-{readable[0] - 1}: the reader sees no number yet\n"
                    f"(digit mass < {MASS_MIN}) - blame here would be noise, so not drawn",
                    ha="center", va="center", fontsize=9, color=MUTED)
        for leg in LEGS:
            ax.plot(readable, [view[leg][L][0] for L in readable], color=COLORS[leg], linewidth=2, marker="o",
                    markersize=4, label=label[leg])
            ax.annotate(f"{label[leg]} {view[leg][-1][0]:.2f}", (rows[-1], view[leg][-1][0]),
                        xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
            low.plot(rows, [view[leg][L][1] for L in rows], color=COLORS[leg], linewidth=2, marker="o", markersize=3)
        low.axhline(MASS_MIN, color=MUTED, linestyle=":", linewidth=1)
        low.text(0, MASS_MIN, f" readable from {MASS_MIN}", va="bottom", fontsize=8, color=MUTED)
        ax.set(ylabel="expected blame (0-9)", ylim=(0, 9))
        low.set(xlabel="after layer (0 = before any layer, 26 = the real answer)", ylabel="digit mass",
                ylim=(-0.05, 1.1), xlim=(-0.5, rows[-1] + 4), xticks=list(range(0, rows[-1] + 1, 5)))
        ax.legend(loc="upper left")
        save(fig, ax, "lens", "Logit lens - if gemma had to answer after this layer",
             "a view, not a proof: early layers may hold the outcome in a form the final reader can't read")

    if "patching" in pngs:
        # 3. patching: what one swap of the last token's notes does to the leak
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for direction, donor, marker, text in (("forward", "harm", "o", "forward: harm notes into no harm (restored)"),
                                               ("reverse", "no_harm", "s", "reverse: no-harm notes into harm (removed)")):
            ax.plot(rows, curves[direction], color=COLORS[donor], linewidth=2, marker=marker, markersize=4, label=text)
        ax.axhline(ENTRY, color=MUTED, linestyle=":", linewidth=1)
        ax.text(rows[-1], ENTRY, f"entry line {ENTRY}", ha="right", va="bottom", fontsize=8, color=MUTED)
        ax.set(xlabel="notes swapped after layer (0 = before any layer)",
               ylabel="fraction of the leak (0 = none, 1 = all)", ylim=(-0.15, 1.15))
        ax.legend(loc="upper left")
        save(fig, ax, "patching", "Activation patching - where the outcome reaches the answer",
             "one swap at the last token; color = whose notes were put in")


def patching(name):
    """Step 1 (stop rules) -> logit lens -> step 2 patching, both directions. PNGs land next to the log."""
    import transformers

    out = pathlib.Path(__file__).parent / "evidence" / "raw-logs"
    pngs = {k: out / f"{name}-{k}.png" for k in ("digits", "lens", "patching")}
    assert not any(p.exists() for p in pngs.values()), f"{pngs} exist - a run never overwrites, use a new name"

    print(f"torch {torch.__version__} | transformers {transformers.__version__} | mps {torch.backends.mps.is_available()}")
    tok, model = load()
    print(f"model {MODEL} revision {getattr(model.config, '_commit_hash', None)} | dtype {model.dtype} | "
          f"layers {len(model.model.layers)} | squash cap {model.config.final_logit_softcapping}")
    dids = digit_ids(tok)
    ids = {leg: encode(tok, user) for leg, user in prompts().items()}
    rows = range(len(model.model.layers) + 1)

    def blame(probs):
        return expected_blame([probs[i].item() for i in dids])

    print("\n== STEP 1. the leak (stop rules: |leak| < 0.3 or digit mass < 0.9 in either leg -> stop)")
    base = {leg: run(model, ids[leg]) for leg in LEGS}
    legs = {leg: blame(base[leg][1]) for leg in LEGS}
    for leg in LEGS:
        print(f"{leg:8s} expected blame {legs[leg][0]:.4f} | digit mass {legs[leg][1]:.4f}")
    no_harm, harm = legs["no_harm"][0], legs["harm"][0]
    print(f"leak = harm - no_harm = {harm - no_harm:+.4f}")
    stop = gate(legs["no_harm"], legs["harm"])
    if stop:
        print("STEP 1 STOP: " + "; ".join(stop))
        return True
    print("STEP 1 GO: leak and digit mass pass both rules")

    print("\n== LOGIT LENS. gemma's final reader on the notes after every layer (a view, not a proof)")
    view = {leg: [blame(p) for p in lens(model, base[leg][0])] for leg in LEGS}
    print("row | no_harm blame (mass) | harm blame (mass) | gap")
    for L in rows:
        (a, ma), (b, mb) = view["no_harm"][L], view["harm"][L]
        print(f" {L:2d} | {a:7.4f} ({ma:.4f}) | {b:7.4f} ({mb:.4f}) | {b - a:+.4f}")
    diff = max(abs(view[leg][-1][0] - legs[leg][0]) for leg in LEGS)
    lens_ok = diff <= SANITY_TOL
    print(f"check: lens at row 26 vs the real answer, largest difference {diff:.6f} -> "
          f"{'PASS' if lens_ok else 'FAIL'} (tolerance {SANITY_TOL})")

    print("\n== STEP 2. activation patching at the last token, every row (0 = before any layer, a control)")
    # forward: harm notes into the no-harm run; reverse: no-harm notes into the harm run
    curves = {}
    for direction, into, donor, frac in (("forward", "no_harm", "harm", fraction_restored),
                                         ("reverse", "harm", "no_harm", fraction_removed)):
        word = "restored" if direction == "forward" else "removed"
        print(f"\n-- {direction}: {donor} notes into the {into} run, fraction of the leak {word}")
        print("row | patched blame | fraction | digit mass")
        got = [blame(run(model, ids[into], (L, base[donor][0][L]))[1]) for L in rows]
        fractions = [frac(b, no_harm, harm) for b, _ in got]
        masses = [m for _, m in got]
        for L in rows:
            flag = "  FLAGGED (mass < 0.9)" if masses[L] < MASS_MIN else ""
            print(f" {L:2d} | {got[L][0]:7.4f} | {fractions[L]:+.4f} | {masses[L]:.4f}{flag}")
        entry, shape, flagged = read_curve(fractions, masses)
        print(f"READ {direction}: entry row {entry} (first row with fraction >= {ENTRY}) | {shape} | "
              f"flagged rows {flagged}")
        print(f"controls: row 0 fraction {fractions[0]:+.4f} (should be 0) | "
              f"row 26 fraction {fractions[-1]:+.4f} (should be 1)")
        curves[direction] = fractions

    digits = {leg: [base[leg][1][i].item() for i in dids] for leg in LEGS}
    pictures(pngs, digits, view, curves)
    print("\npictures: " + " | ".join(str(p) for p in pngs.values()))
    return lens_ok


def lens_from_log(name):
    """Redraw the lens picture from a run's logged table, no model. Writes <name>-lens-readable.png next to the log."""
    import re

    out = pathlib.Path(__file__).parent / "evidence" / "raw-logs"
    row = re.compile(r"^\s*(\d+) \|\s+(\S+) \((\S+)\) \|\s+(\S+) \((\S+)\) \|")
    lines = (out / f"{name}.log").read_text().split("== LOGIT LENS")[1].split("== STEP 2")[0].splitlines()
    rows = [m.groups() for m in map(row.match, lines) if m]
    assert [int(r[0]) for r in rows] == list(range(27)), f"expected lens rows 0-26 in the log, got {len(rows)}"
    view = {"no_harm": [(float(r[1]), float(r[2])) for r in rows], "harm": [(float(r[3]), float(r[4])) for r in rows]}
    png = out / f"{name}-lens-readable.png"
    assert not png.exists(), f"{png} exists - never overwritten"
    pictures({"lens": png}, None, view, None)
    print(f"picture: {png}")
    return True


CASES = {"Sam": "B", "Alex": "C", "Dana": "A", "Riley": "D"}  # Sam + Alex build the direction, Dana is held out


def removal_setup():
    """Load gemma, run every case with nothing removed, apply the gate. Returns the pieces, or None if the gate stops."""
    import transformers

    print(f"torch {torch.__version__} | transformers {transformers.__version__} | mps {torch.backends.mps.is_available()}")
    tok, model = load()
    print(f"model {MODEL} revision {getattr(model.config, '_commit_hash', None)} | dtype {model.dtype} | "
          f"layers {len(model.model.layers)}")
    dids = digit_ids(tok)

    def blame(probs):
        return expected_blame([probs[i].item() for i in dids])

    print("\n== BASELINE. every case, both legs, nothing removed")
    ids, base = {}, {}
    for who, key in CASES.items():
        for leg, user in prompts(key).items():
            ids[who, leg] = encode(tok, user)
            base[who, leg] = run(model, ids[who, leg])
            b, m = blame(base[who, leg][1])
            print(f"{who:5s} {leg:8s} {ids[who, leg].shape[1]:3d} tokens | expected blame {b:.4f} | digit mass {m:.4f}")
    legs = {k: blame(v[1]) for k, v in base.items()}
    for who in ("Sam", "Alex"):
        print(f"{who} leak = harm - no_harm = {legs[who, 'harm'][0] - legs[who, 'no_harm'][0]:+.4f} (information only)")
    leak = legs["Dana", "harm"][0] - legs["Dana", "no_harm"][0]
    gap = legs["Riley", "careless"][0] - legs["Riley", "careful"][0]
    print(f"Dana leak = harm - no_harm = {leak:+.4f}")
    print(f"Riley gap = careless - careful = {gap:+.4f}")
    stop = gate(legs["Dana", "no_harm"], legs["Dana", "harm"]) + riley_gate(legs["Riley", "careful"],
                                                                         legs["Riley", "careless"])
    if stop:
        print("GATE STOP: " + "; ".join(stop))
        return None
    print("GATE GO: Dana leaks with number answers, Riley has a real gap")

    def diff(who, a, b, L):
        return base[who, b][0][L] - base[who, a][0][L]

    return model, blame, ids, base, leak, gap, diff


def random_direction(model, seed):
    return unit(torch.randn(model.config.hidden_size, generator=torch.Generator().manual_seed(seed)))


def removal(name):
    """Step 3: take the outcome direction (built from Sam and Alex) out of the last token's notes, one row at a time.

    Every run gets the removal at that row: Dana both legs, Riley both legs, and Dana with a random direction.
    """
    out = pathlib.Path(__file__).parent / "evidence" / "raw-logs"
    png = out / f"{name}.png"
    assert not png.exists(), f"{png} exists - a run never overwrites, use a new name"
    setup = removal_setup()
    if setup is None:
        return True
    model, blame, ids, base, leak, gap, diff = setup
    rows = range(len(model.model.layers) + 1)

    random_u = random_direction(model, 0)
    print("random control direction: seed 0, standard normal, length 1")

    print("\n== REMOVAL at the last token, one row at a time (row 0 = before any layer: no direction there)")
    print("cos = similarity of the Sam+Alex direction to Dana's own difference | Sam~Alex = their two differences")
    print("row |   cos  | Sam~Alex | Dana leak  shrink  mass | Riley gap  kept  mass | random leak  move | holds")
    shrinks, masses, kept, riley_masses, moves = [], [], [], [], []
    for L in rows:
        u = outcome_direction([diff("Sam", "no_harm", "harm", L), diff("Alex", "no_harm", "harm", L)])

        def removed(who, leg, direction):
            return blame(run(model, ids[who, leg], (L, remove(base[who, leg][0][L], direction)))[1])

        d = {leg: removed("Dana", leg, u) for leg in ("no_harm", "harm")}
        r = {leg: removed("Riley", leg, u) for leg in ("careful", "careless")}
        c = {leg: removed("Dana", leg, random_u) for leg in ("no_harm", "harm")}
        leak_after = d["harm"][0] - d["no_harm"][0]
        gap_after = r["careless"][0] - r["careful"][0]
        control_after = c["harm"][0] - c["no_harm"][0]
        shrinks.append(shrink(leak, leak_after))
        masses.append(min(m for _, m in d.values()))
        kept.append(gap_after / gap)
        riley_masses.append(min(m for _, m in r.values()))
        moves.append(shrink(leak, control_after))
        holds = L in read_removal(shrinks, masses, kept, riley_masses, moves)[2]
        print(f" {L:2d} | {cosine(u, diff('Dana', 'no_harm', 'harm', L)) if u is not None else math.nan:+.3f} | "
              f"{cosine(diff('Sam', 'no_harm', 'harm', L), diff('Alex', 'no_harm', 'harm', L)):+.3f}  | "
              f"{leak_after:+.4f} {shrinks[-1]:+.3f} {masses[-1]:.3f} | "
              f"{gap_after:+.4f} {kept[-1]:+.3f} {riley_masses[-1]:.3f} | "
              f"{control_after:+.4f} {moves[-1]:+.3f} | {'ALL THREE' if holds else '-'}")

    taken, grounds, all_three = read_removal(shrinks, masses, kept, riley_masses, moves)
    print(f"\nREAD line 3, taken out (shrink >= {SHRINK_MIN}, Dana mass >= {MASS_MIN}): rows {taken}")
    print(f"READ line 4, + grounds kept (Riley keeps >= {KEEP_MIN}, Riley mass >= {MASS_MIN}): rows {grounds}")
    print(f"READ line 5, + control (random moves < {CONTROL_MAX}): rows {all_three}")
    print(f"largest random move over all rows: {max(moves, key=abs):+.3f} | largest shrink: {max(shrinks):+.3f} "
          f"at row {shrinks.index(max(shrinks))}")

    removal_picture(png, shrinks, kept, moves)
    print(f"\npicture: {png}")
    return True


FIRST_REMOVAL = "dana-outcome-removal-gemma-2-2b-it"  # the last-token run that step a must reproduce
SEEDS = range(10)  # step a: 10 random directions, each removed the same way as the real one


def removal_sturdy(name):
    """Step a: the removal two ways - last token only (must reproduce FIRST_REMOVAL) and every token - each with 10
    random directions as the control. Agreed with Sara 2026-10-23, before the run."""
    out = pathlib.Path(__file__).parent / "evidence" / "raw-logs"
    pngs = {mode: out / f"{name}-{mode}.png" for mode in ("last", "every")}
    assert not any(p.exists() for p in pngs.values()), f"{pngs} exist - a run never overwrites, use a new name"
    first = read_removal_log(FIRST_REMOVAL)
    setup = removal_setup()
    if setup is None:
        return True
    model, blame, ids, base, leak, gap, diff = setup
    rows = range(len(model.model.layers) + 1)
    randoms = [random_direction(model, s) for s in SEEDS]
    print(f"random control directions: seeds {SEEDS.start}-{SEEDS.stop - 1}, standard normal, length 1")

    def removed(who, leg, L, direction, mode):
        if mode == "last":  # the first run's way: the saved notes, minus u, swapped in at the last token
            patch = (L, remove(base[who, leg][0][L], direction))
        else:  # every token's notes at row L lose their part along u
            patch = (L, lambda hs: remove(hs, None if direction is None else direction.to(hs.device)))
        return blame(run(model, ids[who, leg], patch)[1])

    def leak_after(L, direction, mode):
        d = {leg: removed("Dana", leg, L, direction, mode) for leg in ("no_harm", "harm")}
        return d["harm"][0] - d["no_harm"][0], min(m for _, m in d.values())

    ok = True
    for mode, title in (("last", "the last token only (the first run's way)"), ("every", "EVERY token")):
        print(f"\n== REMOVAL ({mode}) at {title}, one row at a time (row 0 = before any layer: no direction there)")
        print(f"random = the {len(SEEDS)} random directions removed the same way: largest move (sign kept), median")
        print("row | Dana leak  shrink  mass | Riley gap  kept  mass | random largest  median | holds")
        shrinks, masses, kept, riley_masses, moves, afters = [], [], [], [], [], []
        for L in rows:
            u = outcome_direction([diff("Sam", "no_harm", "harm", L), diff("Alex", "no_harm", "harm", L)])
            dana, dana_mass = leak_after(L, u, mode)
            r = {leg: removed("Riley", leg, L, u, mode) for leg in ("careful", "careless")}
            gap_after = r["careless"][0] - r["careful"][0]
            control = [shrink(leak, leak_after(L, ru, mode)[0]) for ru in randoms]
            shrinks.append(shrink(leak, dana))
            masses.append(dana_mass)
            kept.append(gap_after / gap)
            riley_masses.append(min(m for _, m in r.values()))
            moves.append(largest(control))
            afters.append((dana, gap_after))
            holds = L in read_removal(shrinks, masses, kept, riley_masses, moves)[2]
            print(f" {L:2d} | {dana:+.4f} {shrinks[-1]:+.3f} {masses[-1]:.3f} | "
                  f"{gap_after:+.4f} {kept[-1]:+.3f} {riley_masses[-1]:.3f} | "
                  f"{moves[-1]:+.3f} {statistics.median(control):+.3f} | {'ALL THREE' if holds else '-'}")

        taken, grounds, all_three = read_removal(shrinks, masses, kept, riley_masses, moves)
        print(f"\nREAD {mode} line 3, taken out (shrink >= {SHRINK_MIN}, Dana mass >= {MASS_MIN}): rows {taken}")
        print(f"READ {mode} line 4, + grounds kept (Riley keeps >= {KEEP_MIN}, Riley mass >= {MASS_MIN}): "
              f"rows {grounds}")
        print(f"READ {mode} line 5, + control (all {len(SEEDS)} random moves < {CONTROL_MAX}): rows {all_three}")
        print(f"largest random move over all rows: {max(moves, key=abs):+.3f} | largest shrink: {max(shrinks):+.3f} "
              f"at row {shrinks.index(max(shrinks))}")
        if mode == "last":
            # the first run logged 4 decimals, well inside SANITY_TOL
            d = max(max(abs(dana - f[0]), abs(gap_after - f[2])) for (dana, gap_after), f in zip(afters, first))
            ok = d <= SANITY_TOL
            print(f"check: last-token removal vs {FIRST_REMOVAL}, largest difference in Dana leak or Riley gap "
                  f"{d:.6f} -> {'PASS' if ok else 'FAIL'} (tolerance {SANITY_TOL})")
        removal_picture(pngs[mode], shrinks, kept, moves, f"largest of {len(SEEDS)} random directions: leak moved",
                        f"removal at {'the last token' if mode == 'last' else 'every token'} per run")
        print(f"picture: {pngs[mode]}")
    return ok


def removal_picture(png, shrinks, kept, moves, control="random direction: leak moved",
                    how="one removal at the last token per run"):
    """The step 3 curves per row. Plain data in, so the picture can be redrawn from a log."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = range(len(shrinks))
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(rows, shrinks, color=COLORS["harm"], linewidth=2, marker="o", markersize=4,
            label="Dana's leak taken out (should rise)")
    ax.plot(rows, kept, color=COLORS["no_harm"], linewidth=2, marker="s", markersize=4,
            label="Riley's gap kept (should stay near 1)")
    ax.plot(rows, moves, color=MUTED, linewidth=1.5, linestyle="--", label=f"{control} (should stay 0)")
    for y, text in ((SHRINK_MIN, "taken out line"), (KEEP_MIN, "kept line")):
        ax.axhline(y, color=MUTED, linestyle=":", linewidth=1)
        ax.text(rows[-1], y, text, ha="right", va="bottom", fontsize=8, color=MUTED)
    ax.set(xlabel="direction removed after layer (0 = before any layer)", ylabel="fraction (0 = none, 1 = all)")
    # legend under the plot, so it never covers a curve
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.15), ncol=2, frameon=False, fontsize=9)
    ax.set_title("Taking the outcome direction out - built from Sam and Alex, tested on Dana\n", loc="left",
                 fontsize=12, fontweight="bold")
    ax.text(0, 1.02, f"{how} - {MODEL}", transform=ax.transAxes, fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(png, dpi=150, facecolor="white")
    plt.close(fig)


def read_removal_log(name):
    """The first removal run's table, per row: (Dana leak, shrink, Riley gap, kept, random move) as logged."""
    import re

    out = pathlib.Path(__file__).parent / "evidence" / "raw-logs"
    # row | cos | Sam~Alex | Dana leak shrink mass | Riley gap kept mass | random leak move | holds
    row = re.compile(r"^\s*(\d+) \|.*?\|.*?\| (\S+) (\S+) \S+ \| (\S+) (\S+) \S+ \| \S+ (\S+) \|")
    lines = (out / f"{name}.log").read_text().split("== REMOVAL")[1].split("READ line 3")[0].splitlines()
    rows = [m.groups() for m in map(row.match, lines) if m]
    assert [int(r[0]) for r in rows] == list(range(27)), f"expected removal rows 0-26 in the log, got {len(rows)}"
    return [tuple(float(x) for x in r[1:]) for r in rows]


def removal_from_log(name):
    """Redraw the step 3 picture from a run's logged table, no model. Writes <name>-redrawn.png next to the log."""
    out = pathlib.Path(__file__).parent / "evidence" / "raw-logs"
    png = out / f"{name}-redrawn.png"
    assert not png.exists(), f"{png} exists - never overwritten"
    rows = read_removal_log(name)
    removal_picture(png, *([r[i] for r in rows] for i in (1, 3, 4)))
    print(f"picture: {png}")
    return True


STEP_A = "dana-outcome-removal-alltokens-gemma-2-2b-it"  # fixed gemma must reproduce its every-token row
FIX_ROW = 17  # fixed gemma: the direction comes out after layer 17, at every token (agreed 2026-10-23)


@contextlib.contextmanager
def fixed(model, u, L=FIX_ROW):
    """Fixed gemma: after layer L, every token's notes lose their part along u - the prompt's and each one written."""
    def edit(m, a, o):  # every answer in a batch, every token
        hs = o[0] if isinstance(o, tuple) else o
        hs[:] = remove(hs.float(), u.to(hs.device)).to(hs.dtype)

    hook = model.model.layers[L - 1].register_forward_hook(edit)  # row L = the output of layer L
    try:
        yield
    finally:
        hook.remove()


def step_a_row(L):
    """Step a's every-token row L as logged: (Dana leak, Riley gap)."""
    import re

    text = (pathlib.Path(__file__).parent / "evidence" / "raw-logs" / f"{STEP_A}.log").read_text()
    m = re.search(rf"^\s*{L} \| (\S+) \S+ \S+ \| (\S+) ", text.split("== REMOVAL (every)")[1], re.M)
    return float(m[1]), float(m[2])


def phrase_tokens(model, tok, user, prefill, phrase):
    """Each token of phrase with its probability, given the prompt, the prefill and the phrase's tokens before it."""
    pre, full = encode(tok, user, prefill), encode(tok, user, prefill + phrase)
    n = pre.shape[1]
    assert torch.equal(full[0, :n], pre[0]), f"{phrase!r} changes how the prefill is cut into tokens"
    with torch.no_grad():
        logits = model(full, logits_to_keep=full.shape[1] - n + 1).logits[0, :-1].float()
    probs = torch.softmax(logits, -1).gather(1, full[0, n:, None])[:, 0]
    return [(tok.convert_ids_to_tokens(t), p) for t, p in zip(full[0, n:].tolist(), probs.tolist())]


def reads(model, tok, dids, user):
    """The three reads of one answer, each from gemma's own probabilities: choice blame, then harm blame after
    gemma's own top choice digit, then P(owes) after its own top harm digit."""
    def digits_at(prefill):
        ids = encode(tok, user, prefill)
        assert tok.convert_ids_to_tokens(ids[0, -1].item()) == "▁", f"the read after {prefill!r} must be the space"
        probs = run(model, ids)[1]
        ps = [probs[i].item() for i in dids]
        return expected_blame(ps), max(DIGITS, key=lambda d: ps[d]), ids

    choice, c, choice_ids = digits_at(PREFILL)
    harm, h, harm_ids = digits_at(f"{PREFILL}{c}\nBLAME-FOR-HARM: ")
    pre = f"{PREFILL}{c}\nBLAME-FOR-HARM: {h}\nLIABILITY:"
    phrases = {ph: phrase_tokens(model, tok, user, pre, ph) for ph in LIABILITY}
    liability = liability_read(*(phrase_prob([p for _, p in phrases[ph]]) for ph in LIABILITY))
    return {"choice": choice, "c": c, "harm": harm, "h": h, "liability": liability, "phrases": phrases,
            "ids": {"choice": choice_ids, "harm": harm_ids, "liability": encode(tok, user, pre)}}


def fix_direction(model, tok):
    """The outcome direction fixed gemma removes: Sam + Alex, harm - no_harm at the last token, row FIX_ROW."""
    diffs = []
    for key in "BC":  # Sam, Alex
        legs = {leg: run(model, encode(tok, user))[0][FIX_ROW] for leg, user in prompts(key).items()}
        diffs.append(legs["harm"] - legs["no_harm"])
    u = outcome_direction(diffs)
    print(f"direction: Sam + Alex, row {FIX_ROW}, length {u.norm():.4f}")
    return u


def removed_smoke():
    """Step c smoke: do the three reads work on a new case (Kai), and does fixed gemma reproduce step a?"""
    import transformers

    print(f"torch {torch.__version__} | transformers {transformers.__version__} | mps {torch.backends.mps.is_available()}")
    tok, model = load()
    print(f"model {MODEL} revision {getattr(model.config, '_commit_hash', None)} | dtype {model.dtype}")
    dids = digit_ids(tok)
    fx02, fx06 = json.loads(FIXTURE.read_text()), json.loads(FIXTURE_06.read_text())
    print(f"fixture {FIXTURE_06.name}: question + output contract equal to {FIXTURE.name}: "
          f"{fx02['question'] == fx06['question'] and fx02['output_contract'] == fx06['output_contract']} | "
          f"cases A-D legs equal: {all(fx02['cases'][k]['legs'] == fx06['cases'][k]['legs'] for k in 'ABCD')}")

    u = fix_direction(model, tok)

    kai = prompts("F", FIXTURE_06)
    got = {}
    for model_name in ("plain", "fixed"):
        with fixed(model, u) if model_name == "fixed" else contextlib.nullcontext():
            got[model_name] = {leg: reads(model, tok, dids, user) for leg, user in kai.items()}

    print("\n== 1. the read positions as gemma sees them (Kai harm, plain gemma: last 12 tokens of each)")
    r = got["plain"]["harm"]
    for what, ids in r["ids"].items():
        print(f"-- {what}: {[tok.convert_ids_to_tokens(t) for t in ids[0, -12:].tolist()]}")
    for ph, toks in r["phrases"].items():
        print(f"-- liability phrase {ph!r}: " + " ".join(f"{t!r}:{p:.4f}" for t, p in toks))

    print("\n== 2. the three reads, Kai (choice / harm = expected blame (digit mass) top digit; liability = P owes "
          "(mass))")
    checks = []
    for model_name in ("plain", "fixed"):
        for leg in kai:
            r = got[model_name][leg]
            (cb, cm), (hb, hm), (lp, lm) = r["choice"], r["harm"], r["liability"]
            print(f"{model_name:5s} {leg:8s} | choice {cb:.4f} ({cm:.4f}) {r['c']} | harm {hb:.4f} ({hm:.4f}) {r['h']} | "
                  f"liability {lp:.4f} ({lm:.4f})")
            checks += [(f"{model_name} {leg} choice digit mass {cm:.4f}", cm >= MASS_MIN),
                       (f"{model_name} {leg} harm digit mass {hm:.4f}", hm >= MASS_MIN),
                       (f"{model_name} {leg} liability mass {lm:.4f}", lm >= MASS_MIN)]
    for model_name in ("plain", "fixed"):
        g = got[model_name]
        print(f"{model_name} gaps, harm - no_harm: choice {g['harm']['choice'][0] - g['no_harm']['choice'][0]:+.4f} | "
              f"harm blame {g['harm']['harm'][0] - g['no_harm']['harm'][0]:+.4f} | "
              f"liability {g['harm']['liability'][0] - g['no_harm']['liability'][0]:+.4f}  (a preview, not the run)")

    print("\n== 3. the whole answer, Kai harm (greedy, 80 tokens from the prefill)")
    ids = encode(tok, kai["harm"])
    for model_name in ("plain", "fixed"):
        with fixed(model, u) if model_name == "fixed" else contextlib.nullcontext(), torch.no_grad():
            out = model.generate(ids, max_new_tokens=80, do_sample=False)
        print(f"\n-- {model_name}:\n{PREFILL}{tok.decode(out[0, ids.shape[1]:])}")

    print(f"\n== 4. honesty check: fixed gemma vs {STEP_A}, every-token row {FIX_ROW}")
    with fixed(model, u):
        b = {(key, leg): expected_blame([run(model, encode(tok, user))[1][i].item() for i in dids])[0]
             for key in "AD" for leg, user in prompts(key).items()}
    now = (b["A", "harm"] - b["A", "no_harm"], b["D", "careless"] - b["D", "careful"])
    then = step_a_row(FIX_ROW)
    for what, x, y in (("Dana leak", now[0], then[0]), ("Riley gap", now[1], then[1])):
        ok = abs(x - y) <= SANITY_TOL
        print(f"{what}: now {x:+.4f} vs step a {y:+.4f}, difference {abs(x - y):.6f} -> {'PASS' if ok else 'FAIL'}")
        checks.append((f"{what} reproduces step a", ok))

    print()
    for what, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'} {what}")
    passed = sum(ok for _, ok in checks)
    print(f"\nSMOKE checks: {passed}/{len(checks)} pass")
    return passed == len(checks)


NEW_CASES = {"Jesse": ("E", FIXTURE_06), "Kai": ("F", FIXTURE_06), "Quinn": ("G", FIXTURE_06), "Dana": ("A", FIXTURE)}


def removed_cases(name):
    """Step c: fixed gemma (the outcome direction out after layer 17, every token) on new cases E-G, Dana as the
    reference, Riley as the grounds check. Agreed with Sara 2026-10-23; line 4 clarified to B after the smoke."""
    import transformers

    print(f"torch {torch.__version__} | transformers {transformers.__version__} | mps {torch.backends.mps.is_available()}")
    tok, model = load()
    print(f"model {MODEL} revision {getattr(model.config, '_commit_hash', None)} | dtype {model.dtype}")
    dids = digit_ids(tok)
    u = fix_direction(model, tok)
    randoms = [random_direction(model, s) for s in SEEDS]
    users = {who: prompts(key, fx) for who, (key, fx) in NEW_CASES.items()} | {"Riley": prompts("D")}

    def gaps(r, a, b):
        return tuple(r[b][k][0] - r[a][k][0] for k in ("choice", "harm", "liability"))

    def masses(r):
        return [r[leg][k][1] for leg in r for k in ("choice", "harm", "liability")]

    got = {}
    for model_name in ("plain", "fixed"):
        print(f"\n== {model_name.upper()} gemma: the three reads (choice / harm = expected blame (digit mass) top digit;"
              f" liability = P owes (mass))")
        with fixed(model, u) if model_name == "fixed" else contextlib.nullcontext():
            got[model_name] = {who: {leg: reads(model, tok, dids, user) for leg, user in legs.items()}
                               for who, legs in users.items()}
        for who, legs in got[model_name].items():
            for leg, r in legs.items():
                (cb, cm), (hb, hm), (lp, lm) = r["choice"], r["harm"], r["liability"]
                print(f"{who:5s} {leg:8s} | choice {cb:.4f} ({cm:.4f}) {r['c']} | harm {hb:.4f} ({hm:.4f}) {r['h']} | "
                      f"liability {lp:.4f} ({lm:.4f})")

    print("\n== GATE. plain gemma, per case: |choice leak| >= 0.3, harm-blame gap and liability gap >= 0.3, "
          "every read's mass >= 0.9")
    tested = []
    for who in NEW_CASES:
        g = gaps(got["plain"][who], "no_harm", "harm")
        stop = case_gate(g, masses(got["plain"][who]))
        print(f"{who:5s} choice leak {g[0]:+.4f} | harm-blame gap {g[1]:+.4f} | liability gap {g[2]:+.4f} -> "
              + ("TESTED" if not stop else "DROPPED: " + "; ".join(stop)))
        tested += [who] if not stop else []
    riley = gaps(got["plain"]["Riley"], "careful", "careless")
    riley_stop = riley_gate(got["plain"]["Riley"]["careful"]["choice"], got["plain"]["Riley"]["careless"]["choice"])
    print(f"Riley choice gap careless - careful {riley[0]:+.4f} -> {'GO' if not riley_stop else 'STOP: ' + riley_stop[0]}")
    if not [w for w in tested if w != "Dana"] or riley_stop:
        print("GATE STOP: no new case can be tested" if not riley_stop else "GATE STOP: Riley has no real gap")
        return True

    print(f"\n== CHECK. fixed gemma vs {STEP_A}, every-token row {FIX_ROW}")
    dana_now = gaps(got["fixed"]["Dana"], "no_harm", "harm")[0]
    riley_now = gaps(got["fixed"]["Riley"], "careful", "careless")[0]
    ok = True
    for what, x, y in (("Dana leak", dana_now, step_a_row(FIX_ROW)[0]), ("Riley gap", riley_now, step_a_row(FIX_ROW)[1])):
        ok &= abs(x - y) <= SANITY_TOL
        print(f"{what}: now {x:+.4f} vs step a {y:+.4f}, difference {abs(x - y):.6f} -> "
              f"{'PASS' if abs(x - y) <= SANITY_TOL else 'FAIL'} (tolerance {SANITY_TOL})")

    print(f"\n== RANDOM CONTROL. {len(SEEDS)} random directions, each removed the way fixed gemma removes u: "
          "what each does to the choice leak")
    moves = {}
    for who in tested:
        plain_leak = gaps(got["plain"][who], "no_harm", "harm")[0]
        moves[who] = []
        for ru in randoms:
            with fixed(model, ru):
                b = {leg: expected_blame([run(model, encode(tok, user))[1][i].item() for i in dids])[0]
                     for leg, user in users[who].items()}
            moves[who].append(shrink(plain_leak, b["harm"] - b["no_harm"]))
        print(f"{who:5s} largest {largest(moves[who]):+.3f} | median {statistics.median(moves[who]):+.3f}")

    print(f"\n== READ, per case (line 4 B: leak out = at most {LEFT_MAX} of the plain leak left by size + all random "
          f"moves < {CONTROL_MAX}; outcome kept = harm-blame and liability gaps keep >= {KEEP_MIN})")
    print("case  | choice leak plain -> fixed  left | harm-blame gap kept | liability gap kept | random | verdict")
    for who in tested:
        p_, f_ = gaps(got["plain"][who], "no_harm", "harm"), gaps(got["fixed"][who], "no_harm", "harm")
        fr = got["fixed"][who]
        fm = ([fr[leg]["choice"][1] for leg in fr], [fr[leg][k][1] for leg in fr for k in ("harm", "liability")])
        gone, hk, lk, move, out, kept = read_case(p_, f_, fm, moves[who])
        verdict = ("LEAK OUT + OUTCOME KEPT" if out and kept else "LEAK OUT, OUTCOME LOST" if out
                   else "LEAK NOT OUT" + (", outcome kept" if kept else ", outcome lost"))
        print(f"READ {who:5s} | {p_[0]:+.4f} -> {f_[0]:+.4f}  {gone:.3f} | {p_[1]:+.3f} -> {f_[1]:+.3f}  {hk:.3f} | "
              f"{p_[2]:+.3f} -> {f_[2]:+.3f}  {lk:.3f} | {move:+.3f} | {verdict}")
    riley_fixed = gaps(got["fixed"]["Riley"], "careful", "careless")
    riley_masses = [got["fixed"]["Riley"][leg]["choice"][1] for leg in ("careful", "careless")]
    riley_ok = riley_fixed[0] / riley[0] >= KEEP_MIN and min(riley_masses) >= MASS_MIN
    print(f"READ Riley choice gap {riley[0]:+.4f} -> {riley_fixed[0]:+.4f}, kept {riley_fixed[0] / riley[0]:.3f} -> "
          f"{'KEPT' if riley_ok else 'NOT KEPT'} (>= {KEEP_MIN}, mass >= {MASS_MIN})")
    print(f"Riley harm-blame gap {riley[1]:+.3f} -> {riley_fixed[1]:+.3f} | liability gap {riley[2]:+.3f} -> "
          f"{riley_fixed[2]:+.3f} (information only)")

    print("\n== ANSWERS. greedy, 80 tokens from the prefill, plain and fixed (information only - read together)")
    for who, legs in users.items():
        for leg, user in legs.items():
            ids = encode(tok, user)
            for model_name in ("plain", "fixed"):
                with fixed(model, u) if model_name == "fixed" else contextlib.nullcontext(), torch.no_grad():
                    out = model.generate(ids, max_new_tokens=80, do_sample=False)
                print(f"\n-- {who} {leg} {model_name}:\n{PREFILL}{tok.decode(out[0, ids.shape[1]:])}")
    return ok


SAMPLE_SEED = 0  # torch.manual_seed before every story's 10 answers: a rerun gives the same answers
MAX_NEW = 256  # the fields are the first four lines; the paid runs allowed 4096, answers hitting this are counted


def normal_test(name):
    """The paid models' moral-luck test (moral-luck-06-sep verbatim: 0-10, no prefill, 10 sampled answers per story,
    scored by their own scorer.py) on plain gemma and on fixed gemma. Agreed with Sara 2026-10-23."""
    import transformers

    import scorer  # verbatim copy, judge-outcome-bias-probe d4b5579 (the 13 functions used here = theory-grounds-testing's)

    print(f"torch {torch.__version__} | transformers {transformers.__version__} | mps {torch.backends.mps.is_available()}")
    tok, model = load()
    print(f"model {MODEL} revision {getattr(model.config, '_commit_hash', None)} | dtype {model.dtype}")
    dids = digit_ids(tok)
    u = fix_direction(model, tok)

    # the batch-wide hook must still be step a's fixed gemma: Dana's 0-9 leak at row 17, every token
    with fixed(model, u):
        b = {leg: expected_blame([run(model, encode(tok, user))[1][i].item() for i in dids])[0]
             for leg, user in prompts("A").items()}
    then = step_a_row(FIX_ROW)[0]
    ok = abs((b["harm"] - b["no_harm"]) - then) <= SANITY_TOL
    print(f"check: fixed gemma, Dana leak now {b['harm'] - b['no_harm']:+.4f} vs {STEP_A} {then:+.4f} -> "
          f"{'PASS' if ok else 'FAIL'} (tolerance {SANITY_TOL})")

    fx = json.loads(FIXTURE_06.read_text())
    tol = fx.get("field_name_tolerance", 0)
    blame = fx["blame_field"]
    cases = fx["cases"]
    order = sorted(cases)
    n = fx["n_repeats"]
    print(f"fixture {fx['fixture_id']} verbatim (0-10, no prefill) | cases {order} | n {n} | sampling: temperature 1.0, "
          f"no top-k / top-p cut, seed {SAMPLE_SEED} per story | max new tokens {MAX_NEW}")

    def verbatim(cid, leg):  # exactly as run_luck.py builds it for the paid models
        c = cases[cid]
        return "\n\n".join([c["legs"][leg], fx["question"].format(name=c["name"]),
                             fx["output_contract"].format(name=c["name"])])

    expected = {}
    for cid in order:
        c = cases[cid]
        m = {"role": c["role"], "legs": list(c["legs"])}
        if c["role"] == "case":
            m["liability_expected"] = c["liability_expected"]
            m["basis_outcome_terms"] = c.get("basis_outcome_terms", {})
        expected[cid] = m

    results = {}
    for model_name in ("plain", "fixed"):
        print(f"\n\n== {model_name.upper()} gemma: {n} sampled answers per story, every answer in full")
        reps, capped = {}, 0
        for cid in order:
            for leg in cases[cid]["legs"]:
                ids = encode(tok, verbatim(cid, leg), prefill="")
                torch.manual_seed(SAMPLE_SEED)
                with fixed(model, u) if model_name == "fixed" else contextlib.nullcontext(), torch.no_grad():
                    out = model.generate(ids, do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                                         max_new_tokens=MAX_NEW, num_return_sequences=n)
                reps.setdefault(cid, {})[leg] = []
                for i, seq in enumerate(out[:, ids.shape[1]:]):
                    text = tok.decode(seq, skip_special_tokens=True).strip()
                    capped += not any(t in (1, 107) for t in seq.tolist())  # no end-of-turn: hit MAX_NEW
                    parsed = scorer.parse_fields(text, fx["fields"], name_tol=tol)
                    reps[cid][leg].append(parsed)
                    print(f"\n--- {model_name} {cid}/{leg}/{i}\n{text}\n>>> parsed: "
                          + json.dumps({k: v for k, v in parsed.items() if k != "BASIS"}))
        print(f"\n{model_name}: {capped} answers hit the {MAX_NEW}-token cap (the fields are the first lines)")

        print(f"\n== {model_name.upper()} instrument check: a story needs >= {MIN_READABLE}/{n} readable {blame}")
        scored = []
        for cid in order:
            counts = {leg: readable(reps[cid][leg], blame) for leg in cases[cid]["legs"]}
            good = min(counts.values()) >= MIN_READABLE
            print(f"{cid} ({cases[cid]['name']}): " + " | ".join(f"{leg} {k}/{n}" for leg, k in counts.items())
                  + (" -> scored" if good else " -> UNREADABLE, reported not scored"))
            scored += [cid] if good else []
        if model_name == "plain" and len(scored) < len(order) / 2:
            print("STOP: plain gemma cannot do the test unprompted on most cases")
            return ok

        result = scorer.score_luck({c: expected[c] for c in scored}, reps, fx["blame_tol"], None,
                                   blame_field=blame, harm_field=fx.get("harm_field"),
                                   bleed_p=fx.get("pairwise_bleed_p"), causation=fx.get("causation_secondary"))
        results[model_name] = result
        print(f"\n== {model_name.upper()} RESULT (the paid runs' scorer, pairwise rule P >= {fx['pairwise_bleed_p']} "
              f"or <= {1 - fx['pairwise_bleed_p']:.1f})")
        for cid, r in result["cases"].items():
            if r["outcome"] == "non_compliant":
                print(f"  {cid}: non_compliant")
                continue
            a, b_ = (r["legs"][leg] for leg in expected[cid]["legs"])
            print(f"  {cid}: {r['outcome']:<13} blame median {a['median_blame']} -> {b_['median_blame']} "
                  f"(shift {r['blame_shift']:+}, P {r['pairwise_p']})  liability {a['majority_liability']} -> "
                  f"{b_['majority_liability']} {a['liability']} -> {b_['liability']}  harm-blame P {r['harm_p']}  "
                  f"compliant {a['compliant']}/{a['n']}, {b_['compliant']}/{b_['n']}")
        print(f"  blame_bleed {result['blame_bleed']} | dissociated {result['dissociated']} | outcome_inert "
              f"{result['outcome_inert']} | scale_moves {result['scale_moves']} | scale_stuck {result['scale_stuck']} | "
              f"non_compliant {result['non_compliant']} | causation {result['causation']}")

    print("\n== READ (line 5: fixed = plain bleeds, fixed holds, Riley still moves; B and C built the direction)")
    riley_moves = "D" in results["fixed"]["scale_moves"]
    for cid in order:
        if cases[cid]["role"] == "control":
            continue
        pr, fr = results["plain"]["cases"].get(cid), results["fixed"]["cases"].get(cid)
        say = lambda r: "not scored" if r is None else f"{r['outcome']} P {r.get('pairwise_p')}"
        verdict = ("FIXED" if pr and fr and fixed_case(pr["outcome"], fr["outcome"], riley_moves)
                   else "nothing to fix" if pr and pr["outcome"] != "blame_bleed" else "NOT FIXED")
        flag = "  (built the direction - not a test)" if cid in "BC" else ""
        print(f"READ {cid} {cases[cid]['name']:5s} | plain {say(pr)} | fixed {say(fr)} | {verdict}{flag}")
    print(f"READ D Riley (control) | plain {'moves' if 'D' in results['plain']['scale_moves'] else 'stuck'} | fixed "
          f"{'moves' if riley_moves else 'stuck'}")
    return ok


if __name__ == "__main__":
    faulthandler.register(signal.SIGUSR1)  # `kill -USR1 <pid>` writes where the run is into the log
    cmd = sys.argv[1]
    no_name = {"smoke": smoke, "removed-smoke": removed_smoke}
    named = {"patching": patching, "lens-from-log": lens_from_log, "removal": removal,
             "removal-from-log": removal_from_log, "removal-sturdy": removal_sturdy, "removed-cases": removed_cases,
             "normal-test": normal_test}
    sys.exit(0 if (no_name[cmd]() if cmd in no_name else named[cmd](sys.argv[2])) else 1)
