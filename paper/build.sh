#!/usr/bin/env bash
# Compile the preprint and report its page count.
#
#   paper/build.sh [tectonic-binary]
#
# Builds in a staging directory rather than in place. Tectonic fails with
# "failed to open input file" when any component of the path contains
# parentheses, and this repository lives under "Documents (2)", so an in-place
# build cannot work on this machine.
#
# Tectonic is a single binary and downloads what it needs on first run:
#   https://github.com/tectonic-typesetting/tectonic/releases
# arXiv compiles the source itself, so this is for checking the page limit
# before submitting, not a requirement for it.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
TECTONIC=${1:-${TECTONIC:-tectonic}}
command -v "$TECTONIC" >/dev/null 2>&1 || {
    echo "no tectonic binary: pass its path or set TECTONIC=" >&2; exit 1; }

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

cp "$HERE"/*.tex "$HERE"/*.sty "$HERE"/*.bst "$HERE"/*.bib "$STAGE"/ 2>/dev/null || true
# Figures are referenced as results/figures/*.pdf from the document, so the
# staging directory has to mirror that shape rather than flatten it.
mkdir -p "$STAGE/results/figures"
cp "$HERE"/../results/figures/*.pdf "$STAGE/results/figures/" 2>/dev/null || true

cd "$STAGE"
"$TECTONIC" -X compile main.tex 2>&1 | tail -12

if [ -f main.pdf ]; then
    cp main.pdf "$HERE"/main.pdf
    python - <<'PY'
import fitz
d = fitz.open("main.pdf")
words = sum(len(p.get_text().split()) for p in d)
print("pages: %d   words: %d" % (d.page_count, words))
# The body limit is what matters; references do not count toward it.
PY
else
    echo "no PDF produced" >&2; exit 1
fi
