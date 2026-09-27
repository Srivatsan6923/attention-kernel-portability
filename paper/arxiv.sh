#!/usr/bin/env bash
# Build the arXiv submission tarball, then compile it the way arXiv will.
#
#     paper/arxiv.sh /path/to/tectonic [outdir]
#
# Two arXiv rules matter here.
#
#   * arXiv does not run BibTeX, so the .bbl is generated here and shipped with
#     custom.bib. The check at the end fails if the references come out empty.
#   * arXiv compiles in one flat directory, so the figures are copied next to
#     main.tex and \graphicspath is rewritten in the submitted copy only.
#
# The tarball is then compiled on its own, so a missing file fails here.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
TECTONIC=${1:-${TECTONIC:-tectonic}}
OUT=${2:-$HERE/arxiv}
command -v "$TECTONIC" >/dev/null 2>&1 || [ -x "$TECTONIC" ] || {
    echo "no tectonic binary: pass its path or set TECTONIC=" >&2; exit 1; }

rm -rf "$OUT"; mkdir -p "$OUT/src"

# 1. A normal build, to produce the .bbl.
BBL=$(mktemp -d); trap 'rm -rf "$BBL"' EXIT
mkdir -p "$BBL/paper" "$BBL/site/public/figures"
cp "$HERE"/*.tex "$HERE"/*.sty "$HERE"/*.bst "$HERE"/*.bib "$BBL/paper"/
cp "$HERE"/../site/public/figures/*.pdf "$BBL/site/public/figures/"
(cd "$BBL/paper" && "$TECTONIC" -X compile --keep-intermediates main.tex >/dev/null 2>&1)
[ -f "$BBL/paper/main.bbl" ] || { echo "no main.bbl was produced" >&2; exit 1; }

# 2. The submission itself, in one flat directory.
cp "$HERE"/main.tex "$HERE"/acl.sty "$HERE"/acl_natbib.bst \
   "$HERE"/custom.bib "$OUT/src"/
cp "$BBL/paper/main.bbl" "$OUT/src"/
cp "$HERE"/../site/public/figures/figA_separation.pdf \
   "$HERE"/../site/public/figures/figB_transfer.pdf "$OUT/src"/
# Point \graphicspath at the submission directory, in the copy only.
sed -i \
    -e '/^% The figures are the PDFs the website also serves\.$/d' \
    -e 's|^\\graphicspath{{\.\./site/public/figures/}}|\\graphicspath{{./}}|' \
    "$OUT/src/main.tex"
grep -q '\\graphicspath{{\./}}' "$OUT/src/main.tex" || {
    echo "graphicspath was not rewritten; check main.tex" >&2; exit 1; }
# arXiv publishes the source, so no comment should refer to the repository.
if grep -nE '^\s*%.*(results/|site/|\.py\b|ledger|regenerat)' "$OUT/src"/*.tex; then
    echo "the lines above reference the build tree; drop them from the copy" >&2; exit 1
fi

# 3. Compile the submission alone to catch missing files.
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

# An empty bibliography still prints a References heading. Check that every
# cited key's first-author surname appears in the rendered references.
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
