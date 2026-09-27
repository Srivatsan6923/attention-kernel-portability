# Pinned environment for the benchmark sweeps.
#
# torch is pinned to 2.9, the newest version with a prebuilt flash-attn wheel.
# Building flash-attn from source takes 1-3 h and 8-16 GB of RAM.
#
# The devel base image is needed because FlashInfer compiles kernels at run
# time with nvcc.

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

# The wheels are several GB, so allow retries and a longer timeout.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_RETRIES=10 \
    PIP_DEFAULT_TIMEOUT=120

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

# Allow pip to install into the system Python (PEP 668).
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Each ARG sits above the step that uses it, so changing one version only
# rebuilds the layers after it. torch gets its own layer because it is large.
ARG TORCH_VERSION=2.9.0
RUN pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch==${TORCH_VERSION}

# Pinned because pandas behaviour changes across major versions.
RUN pip install \
        numpy==2.3.5 pandas==3.0.1 pyarrow==25.0.1 scipy==1.17.1 \
        scikit-learn==1.8.0 matplotlib==3.10.8 pytest==8.4.2

# Look the wheel up in the release assets instead of building its filename.
# The name depends on the torch ABI flag, and its version does not always match
# the release tag. A missing combination fails the build here.
ARG FLASH_ATTN_TAG=v2.8.3
ENV FLASH_ATTN_TAG=${FLASH_ATTN_TAG}
COPY env/resolve_flash_attn.py /tmp/resolve_flash_attn.py
RUN python /tmp/resolve_flash_attn.py > /tmp/wheel_url \
    && cat /tmp/wheel_url \
    && pip install "$(cat /tmp/wheel_url)"

# FlashAttention 3 has no published wheel, so it is built here. nvcc does not
# need an H100 present.
#
# Build from a clone. pip's "#subdirectory=hopper" form breaks because setup.py
# reads ../README.md.
#
# Each compile unit uses up to about 10 GB, so cap MAX_JOBS by the container's
# memory limit. `free` inside a container shows the host's memory.
#
# The build targets sm90a only. On other GPUs impls.has() reports P4h-fa3 as
# unsupported. A prebuilt wheel in env/wheels/ skips the compile, which is
# needed on machines without enough memory. The wheel is gitignored.
ARG FA3_MAX_JOBS=16
COPY env/wheels/ /tmp/wheels/
RUN if ls /tmp/wheels/flash_attn_3-*.whl >/dev/null 2>&1; then \
      pip install /tmp/wheels/flash_attn_3-*.whl; \
    else \
      git clone --depth 1 -b ${FLASH_ATTN_TAG} \
        https://github.com/Dao-AILab/flash-attention.git /tmp/fa \
      && cd /tmp/fa/hopper \
      && TORCH_CUDA_ARCH_LIST="9.0a" MAX_JOBS=${FA3_MAX_JOBS} \
         pip install --no-build-isolation . \
      && cd / && rm -rf /tmp/fa; \
    fi && rm -rf /tmp/wheels

# flashinfer-python is a pure-Python wheel that compiles kernels on first use.
ARG FLASHINFER_VERSION=0.6.18
RUN pip install flashinfer-python==${FLASHINFER_VERSION}

# Keep compile caches on the mounted volume so repeats and restarts reuse them.
ENV TORCHINDUCTOR_CACHE_DIR=/data/cache/inductor \
    TRITON_CACHE_DIR=/data/cache/triton \
    FLASHINFER_WORKSPACE_BASE=/data/cache/flashinfer

# Every row records this sha, so the build fails if it is missing.
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

# Fail the build if a kernel library does not import.
RUN python -c "\
import torch, triton, flash_attn, flashinfer, flash_attn_interface; \
print('torch', torch.__version__, 'triton', triton.__version__, \
      'flash_attn', flash_attn.__version__, 'flashinfer', flashinfer.__version__)"

# Import the project code too, and check that every implementation registered.
RUN python -c "\
import akp.impls, akp.run, akp.check, akp.bench, akp.analysis, akp.preflight; \
n = len(akp.impls.IMPLS); \
print('akp ok:', n, 'implementations registered'); \
assert n >= 15, 'expected the full registry, got %d' % n"

CMD ["python", "-m", "akp.run", "--grid", "smoke"]
