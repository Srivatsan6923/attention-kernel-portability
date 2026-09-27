#!/usr/bin/env sh
# List the GPU architectures each installed library was compiled for.
#
#   docker run --rm <image> sh /workspace/scripts/provenance.sh
#
# Needs no GPU. cuobjdump reads the binaries from disk.
#
# SASS is native code for that architecture. PTX is compiled by the driver at
# load time. A cubin built for sm_80 also runs on sm_86 and sm_89, so a missing
# sm_86 entry means the library runs code built for another GPU there.
set -u

probe() {
    name=$1
    file=$2
    if [ ! -f "$file" ]; then
        printf '%-24s NOT FOUND\n' "$name"
        return
    fi
    sass=$(cuobjdump --list-elf "$file" 2>/dev/null \
           | grep -oE 'sm_[0-9]+' | sort -uV | tr '\n' ' ')
    ptx=$(cuobjdump --list-ptx "$file" 2>/dev/null \
          | grep -oE 'sm_[0-9]+' | sort -uV | tr '\n' ' ')
    size=$(( $(wc -c < "$file") / 1048576 ))
    printf '%-24s %5dMB  SASS[ %s]  PTX[ %s]\n' \
        "$name" "$size" "${sass:-none }" "${ptx:-none }"
}

py() { python -c "$1" 2>/dev/null; }

echo "# binary provenance -- $(date -u +%Y-%m-%dT%H:%M:%SZ)"
py "import torch;print('# torch', torch.__version__)"
py "import flash_attn;print('# flash_attn', flash_attn.__version__)"
py "import flashinfer;print('# flashinfer', flashinfer.__version__)"
echo

# Use find instead of importing, because importing flash_attn_2_cuda needs
# torch and a GPU driver.
probe flash-attn-2 "$(find / -name 'flash_attn_2_cuda*.so' 2>/dev/null | head -1)"
probe flash-attn-3 "$(find / -path '*flash_attn_3*' -name '*.so' 2>/dev/null | head -1)"
probe libtorch_cuda "$(py "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'lib','libtorch_cuda.so'))")"
probe cudnn-engines "$(find / -name 'libcudnn_engines_precompiled*.so*' 2>/dev/null | head -1)"
probe cublas "$(find / -name 'libcublas.so*' 2>/dev/null | head -1)"

# FlashInfer compiles kernels at run time and ships no cubins.
echo
echo "# flashinfer JITs at run time and ships no cubins; see the JIT cache instead"
