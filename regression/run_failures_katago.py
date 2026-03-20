#!/usr/bin/env python3
"""
run_failures_katago.py - Run 13 first-batch regression tests, identify unexpected
failures in reg_genmove/genmove commands, and query KataGo policy for each.

Usage:
  cd regression && python3 run_failures_katago.py
  python3 run_failures_katago.py --gnugo ../interface/gnugo
"""

import argparse
import re
import subprocess
import sys
import os
from dataclasses import dataclass
from math import isnan
from pathlib import Path

HOME = os.path.expanduser("~")

# Defaults (mirrors check_katago_policy.py)
GNUGO_PATH  = Path(__file__).parent.parent / "interface" / "gnugo"
KATAGO_PATH = Path(f"{HOME}/Code/KataGo/cpp/katago")
MODEL_PATH  = Path(f"{HOME}/Code/KataGo-Models/kata1-b28c512nbt-s12674021632-d5782420041.bin.gz")
CONFIG_PATH = Path(f"{HOME}/Code/KataGo/cpp/configs/gtp_example.cfg")

FIRST_BATCH = [
    "reading.tst", "owl.tst", "ld_owl.tst", "optics.tst", "filllib.tst",
    "atari_atari.tst", "connection.tst", "break_in.tst", "blunder.tst",
    "unconditional.tst", "trevora.tst", "nngs1.tst", "strategy.tst",
]

GTP_COLS = list("ABCDEFGHJKLMNOPQRST")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    tst_name: str
    test_id: int
    command: str   # full command string, e.g. "reg_genmove black"
    cmd_name: str  # just the command name
    color: str     # "black" or "white" (for genmove/reg_genmove)
    sgf_path: str  # relative path from regression dir
    num_moves: int # 9999 if not specified
    pattern: str   # raw pattern string e.g. "[H2|J3]" or "[!L18]*"
    expect_fail: bool
    ignore: bool


# ---------------------------------------------------------------------------
# .tst parser
# ---------------------------------------------------------------------------

def parse_tst_file(tst_path: Path) -> list:
    """Parse a .tst file and return a list of TestCase objects."""
    tst_name = tst_path.name
    tests = []
    current_sgf = None
    current_num_moves = 9999
    pending_id = None
    pending_cmd = None

    with open(tst_path) as f:
        for line in f:
            line = line.rstrip('\n').rstrip('\r')

            # Skip comment lines (but not #? lines)
            if line.startswith('#') and not line.startswith('#?'):
                continue

            # loadsgf line
            m = re.match(r'^loadsgf\s+(\S+)(?:\s+(\d+))?', line)
            if m:
                current_sgf = m.group(1)
                current_num_moves = int(m.group(2)) if m.group(2) else 9999
                pending_id = None
                pending_cmd = None
                continue

            # #? pattern line — completes a pending test
            m = re.match(r'^#\?\s*(.*)', line)
            if m and pending_id is not None:
                pattern = m.group(1).strip()
                expect_fail = bool(re.search(r'\*$', pattern))
                ignore = bool(re.search(r'&$', pattern))
                # Parse color from command
                parts = pending_cmd.split()
                cmd_name = parts[0] if parts else ""
                color = parts[1] if len(parts) > 1 else ""
                tests.append(TestCase(
                    tst_name=tst_name,
                    test_id=pending_id,
                    command=pending_cmd,
                    cmd_name=cmd_name,
                    color=color,
                    sgf_path=current_sgf or "",
                    num_moves=current_num_moves,
                    pattern=pattern,
                    expect_fail=expect_fail,
                    ignore=ignore,
                ))
                pending_id = None
                pending_cmd = None
                continue

            # Test-ID line: "27 reg_genmove white"
            m = re.match(r'^(\d+)\s+(.+)', line)
            if m:
                pending_id = int(m.group(1))
                pending_cmd = m.group(2).strip()
                continue

    return tests


# ---------------------------------------------------------------------------
# GnuGo runner
# ---------------------------------------------------------------------------

def run_gnugo(tst_path: Path, gnugo: Path, level: int) -> dict:
    """Run GnuGo on tst_path, return dict mapping test_id -> actual output."""
    result = subprocess.run(
        [str(gnugo), "--quiet", "--mode", "gtp", f"--level={level}"],
        stdin=open(tst_path),
        capture_output=True,
        text=True,
    )
    outputs = {}
    for line in result.stdout.splitlines():
        line = line.rstrip('\r')
        m = re.match(r'^[=?](\d+)\s*(.*)', line)
        if m:
            outputs[int(m.group(1))] = m.group(2)
    return outputs


# ---------------------------------------------------------------------------
# Pattern matching (mirrors regress.awk exactly)
# ---------------------------------------------------------------------------

def is_unexpected_failure(actual: str, pattern_raw: str) -> bool:
    """Return True iff this test is an unexpected failure."""
    ignore = bool(re.search(r'&$', pattern_raw))
    fail   = bool(re.search(r'\*$', pattern_raw))
    if ignore:
        return False
    inner = re.sub(r'^\[', '', pattern_raw)
    inner = re.sub(r'\][&*]*$', '', inner)
    negate = inner.startswith('!')
    regex  = '^' + (inner[1:] if negate else inner) + '$'
    try:
        matched = bool(re.search(regex, actual))
    except re.error:
        matched = False
    if negate:
        matched = not matched
    return not matched and not fail


# ---------------------------------------------------------------------------
# SGF helpers
# ---------------------------------------------------------------------------

def parse_sgf_board_size(sgf_path: Path) -> int:
    content = open(sgf_path).read()
    m = re.search(r'SZ\[(\d+)\]', content)
    return int(m.group(1)) if m else 19


def parse_sgf_komi(sgf_path: Path) -> float:
    content = open(sgf_path).read()
    m = re.search(r'KM\[([\d.]+)\]', content)
    return float(m.group(1)) if m else 6.5


# ---------------------------------------------------------------------------
# Board-size-aware coordinate helpers
# ---------------------------------------------------------------------------

def sgf_to_gtp(sgf_coord: str, board_size: int) -> str:
    col_idx = ord(sgf_coord[0]) - ord('a')
    row_idx = ord(sgf_coord[1]) - ord('a')
    return f"{GTP_COLS[col_idx]}{board_size - row_idx}"


def gtp_to_flat(vertex: str, board_size: int) -> int:
    vertex = vertex.upper()
    col_idx = GTP_COLS.index(vertex[0])
    row_num = int(vertex[1:])
    return (board_size - row_num) * board_size + col_idx


def flat_to_gtp(idx: int, board_size: int) -> str:
    return f"{GTP_COLS[idx % board_size]}{board_size - idx // board_size}"


def parse_sgf_moves(sgf_path: Path, num_moves: int, board_size: int) -> list:
    content = open(sgf_path).read()
    moves = re.findall(r';([BW])\[([a-s]{2})\]', content)
    return [
        ("black" if c == 'B' else "white", sgf_to_gtp(coord, board_size))
        for c, coord in moves[:num_moves]
    ]


def build_gtp_session(moves: list, komi: float, board_size: int) -> str:
    lines = [f"boardsize {board_size}", f"komi {komi}", "clear_board"]
    for color, vertex in moves:
        lines.append(f"play {color} {vertex}")
    lines += ["kata-raw-nn 0", "quit"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# KataGo runner
# ---------------------------------------------------------------------------

def run_katago(gtp_input: str, katago: Path, model: Path, config: Path) -> tuple:
    proc = subprocess.run(
        [str(katago), "gtp", "-model", str(model), "-config", str(config)],
        input=gtp_input, capture_output=True, text=True, timeout=300,
    )
    return proc.stdout, proc.stderr


def parse_policy(stdout: str, board_size: int) -> tuple:
    """Return (policy_flat, pass_prob) for arbitrary board size."""
    lines = stdout.splitlines()
    policy_start = None
    for i, line in enumerate(lines):
        if line.strip() == "policy":
            policy_start = i + 1
    if policy_start is None:
        return None, None
    rows = []
    i = policy_start
    while i < len(lines) and len(rows) < board_size:
        vals = lines[i].split()
        if len(vals) == board_size:
            rows.append([float(v) if v != 'NaN' else float('nan') for v in vals])
        i += 1
    if len(rows) != board_size:
        return None, None
    policy_flat = [v for row in rows for v in row]
    pass_prob = None
    for line in lines[i:i + 5]:
        if line.strip().startswith("policyPass"):
            try:
                pass_prob = float(line.strip().split()[1])
            except (IndexError, ValueError):
                pass
            break
    return policy_flat, pass_prob


# ---------------------------------------------------------------------------
# Rank helpers
# ---------------------------------------------------------------------------

def find_rank(vertex: str, ranked: list) -> tuple:
    """Return (rank, prob) for vertex in ranked list, or (None, None)."""
    v = vertex.upper()
    for rank, (prob, rv) in enumerate(ranked, 1):
        if rv.upper() == v:
            return rank, prob
    return None, None


def best_rank_for_pattern(inner: str, ranked: list) -> tuple:
    """Given pattern inner (no brackets, no negate), return best (rank, prob, vertex)."""
    alternatives = inner.split('|')
    best = None
    for alt in alternatives:
        alt = alt.strip()
        rank, prob = find_rank(alt, ranked)
        if rank is not None:
            if best is None or rank < best[0]:
                best = (rank, prob, alt)
    return best  # (rank, prob, vertex) or None


# ---------------------------------------------------------------------------
# .tst patcher
# ---------------------------------------------------------------------------

def patch_tst_file(tst_path: Path, patches: dict):
    """
    patches: dict mapping test_id (int) -> list of new vertex strings to add
    Reads the .tst file, replaces '#? [PATTERN]' with '#? [PATTERN|v1|v2|v3]*'
    for each patched test, writes back in-place.
    """
    lines = open(tst_path).readlines()
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r'^(\d+)\s+', line)
        if m and int(m.group(1)) in patches:
            test_id = int(m.group(1))
            result.append(line)
            i += 1
            # The very next #? line is this test's pattern
            while i < len(lines):
                nline = lines[i]
                pm = re.match(r'^(#\?\s*)\[(.+?)(\][&*]*)\s*$', nline)
                if pm:
                    prefix = pm.group(1)
                    inner  = pm.group(2)
                    existing = set(inner.split('|'))
                    new_verts = [v for v in patches[test_id] if v not in existing]
                    new_inner = inner + ('|' + '|'.join(new_verts) if new_verts else '')
                    result.append(f"{prefix}[{new_inner}]*\n")
                    i += 1
                    break
                result.append(nline)
                i += 1
            continue
        result.append(line)
        i += 1
    open(tst_path, 'w').writelines(result)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gnugo",  default=None, metavar="PATH")
    parser.add_argument("--katago", default=None, metavar="PATH")
    parser.add_argument("--model",  default=None, metavar="PATH")
    parser.add_argument("--config", default=None, metavar="PATH")
    parser.add_argument("--level",  type=int, default=10, metavar="N",
                        help="GnuGo level (default 10)")
    parser.add_argument("--patch", action="store_true",
                        help="Rewrite .tst files in-place to add KataGo alternatives and * marker")
    args = parser.parse_args()

    gnugo  = Path(args.gnugo)  if args.gnugo  else GNUGO_PATH
    katago = Path(args.katago) if args.katago else KATAGO_PATH
    model  = Path(args.model)  if args.model  else MODEL_PATH
    config = Path(args.config) if args.config else CONFIG_PATH

    regression_dir = Path(__file__).parent

    for path, label in [(gnugo, "GnuGo"), (katago, "KataGo"),
                        (model, "Model"), (config, "Config")]:
        if not path.exists():
            print(f"ERROR: {label} not found: {path}", file=sys.stderr)
            sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 1: collect unexpected failures across all 13 test files
    # -----------------------------------------------------------------------
    failures = []

    for tst_name in FIRST_BATCH:
        tst_path = regression_dir / tst_name
        if not tst_path.exists():
            print(f"WARNING: {tst_name} not found, skipping", file=sys.stderr)
            continue

        print(f"Running {tst_name}...", file=sys.stderr)
        test_cases = parse_tst_file(tst_path)
        outputs = run_gnugo(tst_path, gnugo, args.level)

        for tc in test_cases:
            actual = outputs.get(tc.test_id, "")
            if is_unexpected_failure(actual, tc.pattern):
                # Only include genmove/reg_genmove for KataGo analysis
                if tc.cmd_name in ("reg_genmove", "genmove"):
                    failures.append((tc, actual))
                    print(f"  FAIL {tc.test_id}: got '{actual}', expected '{tc.pattern}'",
                          file=sys.stderr)

    print(f"\nFound {len(failures)} unexpected reg_genmove/genmove failures.",
          file=sys.stderr)

    if not failures:
        print("\n| Test Name | Test ID | GnuGo Output | Expected Output | KataGo #1 | KataGo #2 | KataGo #3 |")
        print("|-----------|---------|--------------|-----------------|-----------|-----------|-----------|")
        return

    # -----------------------------------------------------------------------
    # Step 2: query KataGo for each failure
    # -----------------------------------------------------------------------
    rows = []
    patch_map = {}  # (tst_path, test_id) -> [vertex1, vertex2, vertex3]

    for tc, actual_output in failures:
        print(f"KataGo: {tc.tst_name} #{tc.test_id}...", file=sys.stderr)

        sgf_abs = (regression_dir / tc.sgf_path).resolve()
        if not sgf_abs.exists():
            print(f"  WARNING: SGF not found: {sgf_abs}", file=sys.stderr)
            rows.append([tc.tst_name, str(tc.test_id),
                         actual_output, tc.pattern,
                         "SGF missing", "SGF missing", "SGF missing"])
            continue

        board_size = parse_sgf_board_size(sgf_abs)
        komi       = parse_sgf_komi(sgf_abs)
        # loadsgf N means "position before move N": replay N-1 moves
        replay = tc.num_moves - 1 if tc.num_moves != 9999 else 9998

        moves = parse_sgf_moves(sgf_abs, replay, board_size)
        gtp   = build_gtp_session(moves, komi, board_size)

        try:
            stdout, _ = run_katago(gtp, katago, model, config)
        except subprocess.TimeoutExpired:
            rows.append([tc.tst_name, str(tc.test_id),
                         actual_output, tc.pattern,
                         "timeout", "timeout", "timeout"])
            continue

        policy_flat, pass_prob = parse_policy(stdout, board_size)
        if policy_flat is None:
            rows.append([tc.tst_name, str(tc.test_id),
                         actual_output, tc.pattern,
                         "parse error", "parse error", "parse error"])
            continue

        # Build ranked list
        ranked = [(p, flat_to_gtp(i, board_size))
                  for i, p in enumerate(policy_flat) if not isnan(p)]
        if pass_prob is not None:
            ranked.append((pass_prob, "pass"))
        ranked.sort(reverse=True)

        # GnuGo output rank
        gnugo_vertex = actual_output.strip().upper() if actual_output.strip() else ""
        if gnugo_vertex:
            gnugo_rank, gnugo_prob = find_rank(gnugo_vertex, ranked)
            if gnugo_rank is not None:
                gnugo_col = f"{gnugo_vertex} #{gnugo_rank}"
            else:
                gnugo_col = f"{gnugo_vertex} (not found)"
        else:
            gnugo_col = "(empty)"

        # Expected output rank
        inner = re.sub(r'^\[', '', tc.pattern)
        inner = re.sub(r'\][&*]*$', '', inner)
        negate = inner.startswith('!')
        if negate:
            expected_col = f"!{inner[1:]}"
        else:
            best = best_rank_for_pattern(inner, ranked)
            if best is not None:
                rank, prob, vertex = best
                expected_col = f"{vertex} #{rank}"
            else:
                expected_col = inner

        # KataGo top 3
        kata_cols = []
        for prob, vertex in ranked[:3]:
            kata_cols.append(f"{vertex} ({prob * 100:.2f}%)")
        while len(kata_cols) < 3:
            kata_cols.append("—")

        rows.append([tc.tst_name, str(tc.test_id),
                     gnugo_col, expected_col] + kata_cols)

        kata_top3 = [v for _, v in ranked[:3] if v != "pass"][:3]
        tst_path = regression_dir / tc.tst_name
        patch_map[(tst_path, tc.test_id)] = kata_top3

    # -----------------------------------------------------------------------
    # Step 3: print Markdown table
    # -----------------------------------------------------------------------
    headers = ["Test Name", "Test ID", "GnuGo Output", "Expected Output",
               "KataGo #1", "KataGo #2", "KataGo #3"]
    col_widths = [max(len(h), max((len(r[i]) for r in rows), default=0))
                  for i, h in enumerate(headers)]

    def fmt_row(cells):
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, col_widths)) + " |"

    print()
    print(fmt_row(headers))
    print("| " + " | ".join("-" * w for w in col_widths) + " |")
    for row in rows:
        print(fmt_row(row))

    # -----------------------------------------------------------------------
    # Step 4 (optional): patch .tst files in-place
    # -----------------------------------------------------------------------
    if args.patch and patch_map:
        from collections import defaultdict
        by_file = defaultdict(dict)
        for (tst_path, test_id), verts in patch_map.items():
            by_file[tst_path][test_id] = verts
        for tst_path, patches in by_file.items():
            patch_tst_file(tst_path, patches)
            print(f"Patched {tst_path.name}: tests {sorted(patches)}", file=sys.stderr)


if __name__ == "__main__":
    main()
