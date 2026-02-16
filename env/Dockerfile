# Pinned environment for the benchmark sweeps.
#
# torch is pinned to 2.9 because that is the newest release Dao-AILab publishes
# a prebuilt flash-attn wheel for. Building flash-attn from source takes 1-3 h
# and 8-16 GB of RAM per image, which is not worth paying on three platforms.
#
# devel base rather than runtime: FlashInfer JIT-compiles its kernels and needs
# nvcc at run time.

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ARG TORCH_VERSION=2.9.0
ARG FLASH_ATTN_VERSION=2.8.3.post1
ARG FLASHINFER_VERSION=0.2.6

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

# PEP 668: this image is the environment, so installing into it is intended.
ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch==${TORCH_VERSION} \
    && pip install \
        numpy pandas pyarrow scipy scikit-learn matplotlib seaborn pytest

# Derive the wheel name from the interpreter and the installed torch instead of
# hardcoding it: the cxx11 ABI flag tracks how torch itself was built and has
# changed between releases. A wrong guess then fails here rather than silently
# falling back to a source build.
RUN python - <<'PY' > /tmp/wheel_url && pip install "$(cat /tmp/wheel_url)"
import sys, torch
ver = "${FLASH_ATTN_VERSION}"
torch_mm = ".".join(torch.__version__.split("+")[0].split(".")[:2])
abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
py = "cp%d%d" % sys.version_info[:2]
name = (f"flash_attn-{ver}+cu12torch{torch_mm}cxx11abi{abi}"
        f"-{py}-{py}-linux_x86_64.whl")
print(f"https://github.com/Dao-AILab/flash-attention/releases/download/"
      f"v{ver}/{name}")
PY

RUN pip install flashinfer-python==${FLASHINFER_VERSION}

# Compile caches live on the mounted volume so the five process repeats and any
# restart reuse them; Inductor autotune and FlashInfer JIT dominate wall time
# otherwise. scripts/nrp_job.yaml points these at the PVC.
ENV TORCHINDUCTOR_CACHE_DIR=/data/cache/inductor \
    TRITON_CACHE_DIR=/data/cache/triton \
    FLASHINFER_JIT_CACHE_DIR=/data/cache/flashinfer

WORKDIR /workspace
COPY pyproject.toml README.md ./
COPY akp ./akp
COPY tests ./tests
COPY scripts ./scripts
RUN pip install -e .

# Fail the build, not the sweep, if a kernel library did not actually import.
RUN python -c "\
import torch, triton, flash_attn, flashinfer; \
print('torch', torch.__version__, 'triton', triton.__version__, \
      'flash_attn', flash_attn.__version__, 'flashinfer', flashinfer.__version__)"

CMD ["python", "-m", "akp.run", "--grid", "smoke"]
