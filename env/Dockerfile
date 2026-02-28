# Pinned environment for the benchmark sweeps.
#
# torch is pinned to 2.9 because that is the newest release Dao-AILab publishes
# a prebuilt flash-attn wheel for. Building flash-attn from source takes 1-3 h
# and 8-16 GB of RAM per image, which is not worth paying on three platforms.
#
# devel base rather than runtime: FlashInfer JIT-compiles its kernels and needs
# nvcc at run time.

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

# The wheels here are several GB; on a slow link the default single attempt and
# short socket timeout lose a whole layer to one stall.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_RETRIES=10 \
    PIP_DEFAULT_TIMEOUT=120

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

# PEP 668: this image is the environment, so installing into it is intended.
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Each ARG sits directly above the step that uses it: declaring them together at
# the top means bumping any one version invalidates every layer below it.
# torch gets its own layer so a hiccup fetching anything else does not throw
# away three gigabytes of CUDA wheels.
ARG TORCH_VERSION=2.9.0
RUN pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch==${TORCH_VERSION}

# Pinned: analysis.py is written against these, and pandas has changed
# behaviour across majors.
RUN pip install \
        numpy==2.3.5 pandas==3.0.1 pyarrow==25.0.1 scipy==1.17.1 \
        scikit-learn==1.8.0 matplotlib==3.10.8 pytest==8.4.2

# Look the wheel up in the release instead of constructing its filename. The
# cxx11 ABI flag tracks how torch was built, and the version inside the wheel
# name does not always match the release tag (v2.8.3.post1 ships 2.8.3 wheels),
# so a constructed name is a 404 waiting to happen. Resolving it against the
# release means a missing combination fails here, loudly, rather than at
# benchmark time.
ARG FLASH_ATTN_TAG=v2.8.3
ENV FLASH_ATTN_TAG=${FLASH_ATTN_TAG}
COPY env/resolve_flash_attn.py /tmp/resolve_flash_attn.py
RUN python /tmp/resolve_flash_attn.py > /tmp/wheel_url \
    && cat /tmp/wheel_url \
    && pip install "$(cat /tmp/wheel_url)"

# flashinfer-python ships a pure-Python wheel and JIT-compiles kernels on first
# use. Older versions are sdist-only, and pip then builds them in an isolated
# environment that re-downloads torch.
ARG FLASHINFER_VERSION=0.6.18
RUN pip install flashinfer-python==${FLASHINFER_VERSION}

# Compile caches live on the mounted volume so the five process repeats and any
# restart reuse them; Inductor autotune and FlashInfer JIT dominate wall time
# otherwise. scripts/nrp_job.yaml points these at the PVC.
ENV TORCHINDUCTOR_CACHE_DIR=/data/cache/inductor \
    TRITON_CACHE_DIR=/data/cache/triton \
    FLASHINFER_WORKSPACE_BASE=/data/cache/flashinfer

# Every row carries this as its provenance, so a stale or omitted value
# mislabels a whole dataset silently: the first pushed image shipped code from
# one commit while claiming another, and the rows pointed at a commit that
# could not produce them. Fail the build rather than ship that.
#
#   docker build -f env/Dockerfile --build-arg GIT_SHA=$(git rev-parse HEAD) .
ARG GIT_SHA=unknown
RUN test ${#GIT_SHA} -eq 40 || { \
      echo "GIT_SHA must be the full 40-char sha, got '${GIT_SHA}'"; exit 1; }
ENV AKP_GIT_SHA=${GIT_SHA}

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

# And import our own code, which the check above never touched. A stray edit
# left a syntax error in the vendored Triton kernel; the build stayed green
# because nothing here imported akp, and every pod would have died on startup
# instead. Importing the registry also proves every implementation registered.
RUN python -c "\
import akp.impls, akp.run, akp.check, akp.bench, akp.analysis, akp.preflight; \
n = len(akp.impls.IMPLS); \
print('akp ok:', n, 'implementations registered'); \
assert n >= 15, 'expected the full registry, got %d' % n"

CMD ["python", "-m", "akp.run", "--grid", "smoke"]
