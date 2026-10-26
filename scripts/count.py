#!/usr/bin/env python3
"""Count a term in files and print every hit - the receipt for any "X is (not) in the source" claim.

An errored search prints nothing, and nothing reads as "0 hits". This script makes the two
impossible to confuse: every failure (bad regex, missing path, unreadable file, zero files
scanned) exits non-zero with a message, and a real result always ends with a count line.

Usage: scripts/count.py <regex> <path> [<path> ...]    (case-insensitive; dirs are walked)
Example: scripts/count.py 'interpretab' papers/
"""
import os
import re
import sys

SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules"}
EXTS = (".md", ".txt", ".py", ".log", ".json", ".jsonl", ".sh")


def files_under(path):
    if os.path.isfile(path):
        yield path
        return
    for d, dirs, names in os.walk(path):
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS)
        for n in sorted(names):
            if n.endswith(EXTS):
                yield os.path.join(d, n)


def main(argv):
    if len(argv) < 3:
        sys.exit(__doc__)
    try:
        pat = re.compile(argv[1], re.IGNORECASE)
    except re.error as e:
        sys.exit(f"SEARCH FAILED: bad regex {argv[1]!r}: {e}")
    missing = [p for p in argv[2:] if not os.path.exists(p)]
    if missing:
        sys.exit(f"SEARCH FAILED: path not found: {', '.join(missing)}")

    scanned, hits, errors = 0, 0, []
    for path in argv[2:]:
        for f in files_under(path):
            try:
                with open(f, encoding="utf-8", newline="") as fh:  # newline="": a progress bar's \r must not split lines
                    lines = fh.read().split("\n")  # so line numbers match grep, the editor and the report
            except (OSError, UnicodeDecodeError) as e:
                errors.append(f"{f}: {e}")
                continue
            scanned += 1
            for i, line in enumerate(lines, 1):
                n = len(pat.findall(line))
                if n:
                    hits += n
                    print(f"{f}:{i}: {'(x%d) ' % n if n > 1 else ''}{line.strip()[:200]}")

    for e in errors:
        print(f"UNREADABLE: {e}", file=sys.stderr)
    if scanned == 0:
        sys.exit("SEARCH FAILED: 0 files scanned - no result, not 0 hits")
    # the scope is part of the receipt (2026-10-13: a 0 over two hand-picked files was reported as "unexplained"; the
    # text was in a file not searched). The count line states what it covers, so quoting it quotes the scope.
    root = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    whole = any(os.path.realpath(p) == root for p in argv[2:])
    scope = "the whole repo" if whole else "ONLY " + ", ".join(argv[2:])
    print(f"== {hits} occurrences of {argv[1]!r} in {scanned} files scanned ({len(errors)} unreadable) - scope: {scope}")
    if hits == 0 and not whole:
        print("== a 0 here supports 'not in these paths' only; for 'not anywhere' or 'source unknown', rerun on .")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main(sys.argv)
