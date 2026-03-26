# GnuGo regression — adversary attack games (v9-vs-atari series)
#
# Game:   147E190062D1616F-L3.sgfs
# Source: https://raw.githubusercontent.com/AlignmentResearch/KataGoVisualizer/main/sgf-viewer/public/sgfs/v9-vs-atari/147E190062D1616F-L3.sgfs
# Paper:  https://arxiv.org/abs/2211.00241 ("Adversarial Policies Beat Superhuman Go AIs")
#
# Covers two independent bugs triggered by the same game:
#
#   Bug 1 — SIGSEGV at n=223 (Patches A-E):
#     Crashes in owl_does_attack() via kworm UB / while-loop UB.
#     Unpatched: SIGSEGV at any level.
#     Patched:   reg_genmove returns G15.
#
#   Bug 2 — OWL hang at n=221 (Method 1 + Method 3):
#     Exponential OWL search triggered at level 9+.
#     Unpatched: hangs indefinitely.
#     Patched:   reg_genmove returns L7 within ~60 s.
#
# Run (from repo root):
#   # Bug 1 only (default level):
#   gnugo-owl-patched --quiet --mode gtp < regression/adversary.tst \
#     | awk -f regression/regress.awk tst=regression/adversary.tst verbose=1
#
#   # Full suite incl. Bug 2 (requires --level 9):
#   gnugo-owl-patched --quiet --level 9 --mode gtp < regression/adversary.tst \
#     | awk -f regression/regress.awk tst=regression/adversary.tst verbose=1

# ─────────────────────────────────────────────────────────────────────────────
# Bug 1 — SIGSEGV smoke test
#   Position: move 223 (White to play after W K4)
#   Unpatched binary crashes here. Patched binary returns G15.
# ─────────────────────────────────────────────────────────────────────────────
loadsgf games/gnugo-crash-K4-B223.sgf 223
1 reg_genmove white
#? [!resign]

# ─────────────────────────────────────────────────────────────────────────────
# Bug 2 — OWL Level 9 hang
#   Position: move 221 (White to play after B C9)
#   Unpatched binary hangs indefinitely at level 9+. Patched returns L7.
#   Note: only reproduces the hang when gnugo is invoked with --level 9 (or 10).
# ─────────────────────────────────────────────────────────────────────────────
loadsgf games/gnugo-crash-K4-B223.sgf 221
2 reg_genmove white
#? [!resign]

# ─────────────────────────────────────────────────────────────────────────────
# KataGo's losing move W D9 at n=222
#   Position: move 222 (White to play after B C9, move 221)
#   KataGo (Victim) played W D9 here, turning a winning position into a loss.
#   Cross-test (GnuGo B vs KataGo W, visits=1):
#     n=222 (before D9): KataGo W wins
#     n=223 (after  D9): KataGo W resigns within 4 moves
#   GnuGo correctly avoids D9 and plays L7 instead.
# ─────────────────────────────────────────────────────────────────────────────
loadsgf games/gnugo-crash-K4-B223.sgf 222
3 reg_genmove white
#? [!D9]
