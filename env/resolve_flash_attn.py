"""Print the flash-attn wheel URL matching the torch in this image.

The wheel a release publishes is keyed on the CUDA major, the torch minor, the
cxx11 ABI flag torch was built with, and the interpreter version. Constructing
that filename by hand is fragile - the version inside the name does not always
match the release tag - so resolve it against the release instead.
"""

import json
import os
import sys
import urllib.request

import torch

tag = os.environ["FLASH_ATTN_TAG"]
url = "https://api.github.com/repos/Dao-AILab/flash-attention/releases/tags/" + tag
assets = json.load(urllib.request.urlopen(url))["assets"]

torch_mm = ".".join(torch.__version__.split("+")[0].split(".")[:2])
abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
want = "cu12torch%scxx11abi%s" % (torch_mm, abi)
py = "cp%d%d" % sys.version_info[:2]
suffix = "-%s-%s-linux_x86_64.whl" % (py, py)

hit = [a for a in assets if want in a["name"] and a["name"].endswith(suffix)]
if not hit:
    sys.exit("no flash-attn wheel in %s for %s %s" % (tag, want, py))
print(hit[0]["browser_download_url"])
