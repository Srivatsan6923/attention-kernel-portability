#!/usr/bin/env sh
# Which GPU architectures did each shipped binary actually get compiled for?
#
#   docker run --rm <image> sh /workspace/scripts/provenance.sh
#
# Needs no GPU and no driver: cuobjdump reads the fat binary off disk. That is
# the point -- the hypothesis is about what the wheels contain, so inspect the
# wheels rather than infer it from timings.
#
# Reading the output: SASS is compiled code for that exact architecture. PTX is
# forward-compatible source the driver JITs at load. A cubin built for sm_80
# also runs on sm_86 and sm_89 -- binary compatibility holds across a minor
# version bump inside one major family -- so an absent sm_86 does not mean the
# library fails there. It means the code running was tuned for a different part.
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

# find(1) rather than importing: flash_attn_2_cuda needs torch loaded first and
# a driver present, neither of which this check should require.
probe flash-attn-2 "$(find / -name 'flash_attn_2_cuda*.so' 2>/dev/null | head -1)"
probe flash-attn-3 "$(find / -path '*flash_attn_3*' -name '*.so' 2>/dev/null | head -1)"
probe libtorch_cuda "$(py "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'lib','libtorch_cuda.so'))")"
probe cudnn-engines "$(find / -name 'libcudnn_engines_precompiled*.so*' 2>/dev/null | head -1)"
probe cublas "$(find / -name 'libcublas.so*' 2>/dev/null | head -1)"

# FlashInfer JITs per shape at run time, so it ships no cubins to inspect; its
# provenance is the JIT cache under FLASHINFER_WORKSPACE_BASE after a sweep.
echo
echo "# flashinfer JITs at run time and ships no cubins; see the JIT cache instead"
