#!/usr/bin/env bash
# Build the arXiv submission tarball, then compile it the way arXiv will.
#
#     paper/arxiv.sh /path/to/tectonic [outdir]
#
# arXiv is not a general build host, and two of its rules bite this paper:
#
#   * It does not run BibTeX. The .bbl has to ship, so this compiles once to
#     produce one and puts it in the tarball. custom.bib ships beside it:
#     leaving it out does not force the .bbl to be used, it just means an
#     engine that does run BibTeX finds no database and silently replaces the
#     bibliography with an empty stub. Both files, and the check below fails
#     if the references come out empty either way.
#   * It compiles in one flat directory. \graphicspath points at
#     ../site/public/figures/, which is right for the repository and wrong
#     here, so the figures are flattened and the line is rewritten in the
#     submitted copy only. paper/main.tex is not modified.
#
# The tarball is then compiled on its own, from a directory holding nothing
# else, so a missing file fails here rather than after upload.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
TECTONIC=${1:-${TECTONIC:-tectonic}}
OUT=${2:-$HERE/arxiv}
command -v "$TECTONIC" >/dev/null 2>&1 || [ -x "$TECTONIC" ] || {
    echo "no tectonic binary: pass its path or set TECTONIC=" >&2; exit 1; }

rm -rf "$OUT"; mkdir -p "$OUT/src"

# 1. A normal build, only to get the .bbl arXiv will not generate itself.
BBL=$(mktemp -d); trap 'rm -rf "$BBL"' EXIT
mkdir -p "$BBL/paper" "$BBL/site/public/figures"
cp "$HERE"/*.tex "$HERE"/*.sty "$HERE"/*.bst "$HERE"/*.bib "$BBL/paper"/
cp "$HERE"/../site/public/figures/*.pdf "$BBL/site/public/figures/"
(cd "$BBL/paper" && "$TECTONIC" -X compile --keep-intermediates main.tex >/dev/null 2>&1)
[ -f "$BBL/paper/main.bbl" ] || { echo "no main.bbl was produced" >&2; exit 1; }

# 2. The submission itself: flat, no build scripts, no Python.
cp "$HERE"/main.tex "$HERE"/tab_avail.tex "$HERE"/acl.sty "$HERE"/acl_natbib.bst \
   "$HERE"/custom.bib "$OUT/src"/
cp "$BBL/paper/main.bbl" "$OUT/src"/
cp "$HERE"/../site/public/figures/figA_separation.pdf \
   "$HERE"/../site/public/figures/figB_transfer.pdf "$OUT/src"/
# Figures sit beside main.tex in a submission, so the repository's path is wrong
# here. Rewrite it in the copy, never in the tracked source, and drop the two
# comment lines above it: arXiv publishes the source, and they describe a
# directory layout that does not exist in what is being read.
sed -i \
    -e '/^% The released figure PDFs are the ones the website serves, and they are$/d' \
    -e '/^% tracked; results\/figures\/ is regenerated output and is not\.$/d' \
    -e 's|^\\graphicspath{{\.\./site/public/figures/}}|\\graphicspath{{./}}|' \
    "$OUT/src/main.tex"
grep -q '\\graphicspath{{\./}}' "$OUT/src/main.tex" || {
    echo "graphicspath was not rewritten; check main.tex" >&2; exit 1; }
# Nothing in the submitted source should point back at the repository tree.
if grep -nE '^\s*%.*(results/|site/|\.py\b|ledger|regenerat)' "$OUT/src"/*.tex; then
    echo "the lines above reference the build tree; drop them from the copy" >&2; exit 1
fi

# 3. Compile the submission alone, so a missing file fails now and not later.
TEST=$(mktemp -d); trap 'rm -rf "$BBL" "$TEST"' EXIT
cp "$OUT/src"/* "$TEST"/
(cd "$TEST" && "$TECTONIC" -X compile main.tex 2>&1 | grep -iE "^error|warning: unre" || true)
[ -f "$TEST/main.pdf" ] || { echo "the tarball does not compile on its own" >&2; exit 1; }
cp "$TEST/main.pdf" "$OUT/main.pdf"

tar -czf "$OUT/arxiv-submission.tar.gz" -C "$OUT/src" .

python - "$OUT" "$HERE" <<'PY'
import os, re, sys
import fitz
out, here = sys.argv[1], sys.argv[2]
d = fitz.open(os.path.join(out, "main.pdf"))
text = "".join(p.get_text() for p in d)
print("compiled from the tarball alone: %d pages, %d words"
      % (d.page_count, len(text.split())))

# An empty bibliography is the failure this build already hit once, and it is
# invisible in the page count: the References heading still prints. Check the
# thing that matters, that every key cited in the source has an entry, by
# looking for each key's first-author surname in the rendered references.
src = open(os.path.join(here, "main.tex"), encoding="utf8").read()
keys = set(k for g in re.findall(r"\\cite[a-z]*\{([^}]*)\}", src)
           for k in g.split(","))
i = text.find("References")
if i < 0:
    sys.exit("no References section in the compiled PDF")
refs = text[i:]
missing = sorted(k for k in keys
                 if re.match(r"[a-z]+", k)
                 and re.match(r"[a-z]+", k).group(0).capitalize() not in refs)
if missing:
    sys.exit("cited but not in the bibliography: %s" % ", ".join(missing))
print("bibliography: all %d cited keys have an entry" % len(keys))

ref = fitz.open(os.path.join(here, "main.pdf"))
if ref.page_count != d.page_count:
    sys.exit("submission is %d pages, the repository PDF is %d"
             % (d.page_count, ref.page_count))
print("matches the repository PDF at %d pages" % ref.page_count)
t = os.path.join(out, "arxiv-submission.tar.gz")
print("tarball: %s (%.0f kB)" % (t, os.path.getsize(t) / 1024))
PY
echo "contents:"
tar -tzf "$OUT/arxiv-submission.tar.gz" | sed 's|^\./||' | grep . | sort | sed 's/^/  /'
