"""Dana outcome leak in gemma-2-2b-it: measure it, then find where the outcome reaches the answer.

Plain explanation: notes.md. Agreed plan and stop rules: next-session-prompt.md.
Top half: the measurement logic (tested by test_outcome_leak.py, no model). Bottom half: the model.

    .venv/bin/python outcome_leak.py smoke              step 0 - look at the machinery, three honesty checks
    .venv/bin/python outcome_leak.py patching <name>    step 1 (stop rules) -> logit lens -> step 2, both directions
    .venv/bin/python outcome_leak.py lens-from-log <name>   redraw the lens picture from that run's log, no model
    .venv/bin/python outcome_leak.py removal <name>     step 3 - take the outcome direction out, Riley and random checks
    .venv/bin/python outcome_leak.py removal-from-log <name>   redraw the step 3 picture from that run's log, no model
    .venv/bin/python outcome_leak.py removal-sturdy <name>   step a - last token vs every token, 10 random directions
"""

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


# ------------------------------------------------------------------------------------------------ the model

import torch  # noqa: E402 - below the logic on purpose: the canned test reads the top half first

MODEL = "google/gemma-2-2b-it"
DEVICE = "mps"
FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "moral-luck-02.json"  # verbatim copy, theory-grounds-testing
PREFILL = "BLAME-FOR-CHOICE: "  # gemma's answer starts with this, so the next token is the number
LEGS = ("no_harm", "harm")
SANITY_TOL = 1e-3  # an honesty check passes if expected blame moves by at most this (fixed before the smoke)


def prompts(key="A"):
    """One case's prompts (A = Dana): leg + question + answer format, joined as moral-luck-02 did, 0-10 -> 0-9."""
    fx = json.loads(FIXTURE.read_text())
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


if __name__ == "__main__":
    faulthandler.register(signal.SIGUSR1)  # `kill -USR1 <pid>` writes where the run is into the log
    cmd = sys.argv[1]
    sys.exit(0 if (smoke() if cmd == "smoke" else {"patching": patching, "lens-from-log": lens_from_log,
                                                   "removal": removal, "removal-from-log": removal_from_log,
                                                   "removal-sturdy": removal_sturdy}[cmd](sys.argv[2])) else 1)
