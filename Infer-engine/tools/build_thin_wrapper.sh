#!/bin/bash
# build_thin_wrapper.sh — Compile thin_wrapper.cpp inside Docker
# Uses nvidia/cuda:12.9.1-devel-ubuntu20.04 (GCC 10.5.0) to match vLLM 0.18.0's ABI.
#
# Usage: bash build_thin_wrapper.sh
# Output: build/libthin_fa2_wrapper.so

set -e
cd /data/fwsr/glm-asr/Infer-engine

mkdir -p build

docker run --rm \
    -v "$(pwd)/cuda/thin_wrapper.cpp:/src/thin_wrapper.cpp:ro" \
    -v "$(pwd)/build:/out" \
    nvidia/cuda:12.9.1-devel-ubuntu20.04 \
    bash -c '
        g++ -shared -fPIC -O2 \
            -o /out/libthin_fa2_wrapper.so \
            /src/thin_wrapper.cpp \
            -I/usr/local/cuda/include
    '

ls -la build/libthin_fa2_wrapper.so
