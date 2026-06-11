cmake_minimum_required(VERSION 3.18)

project(glm_asr_infer LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES OFF)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(CUDA_ROOT "/usr/local/cuda-12.8" CACHE PATH "CUDA toolkit root")
set(GLMASR_CUDA_ARCH "sm_120" CACHE STRING "NVCC -arch target, e.g. sm_120/sm_89/sm_86")
set(
  GLMASR_PYTHON_EXECUTABLE
  "${Python3_EXECUTABLE}"
  CACHE FILEPATH
  "Python interpreter with nvidia-cutlass-dsl and apache-tvm-ffi installed"
)
set(
  GLMASR_EXTRA_RPATH
  ""
  CACHE STRING
  "Optional extra runtime rpath entries"
)

add_executable(
  glm_asr_infer
  include/audio_io.h
  include/cuda_kernels.h
  include/engine.h
  include/kv_pool.h
  include/model.h
  include/safetensors.h
  include/scheduler.h
  include/sequence.h
  include/tokenizer.h
  include/types.h
  third_party/dr_wav.h
  src/main.c
  src/audio_io.c
  src/safetensors.c
  src/tokenizer.c
  src/model.c
  src/engine.c
  src/kv_pool.c
  src/sequence.c
  src/scheduler.c
  cuda/flash_fwd_vllm.c
  cuda/kernels.cu
  cuda/mel.cu
  cuda/encoder.cu
  cuda/decoder.cu
  cuda/nvfp4_cutlass.cu
  cuda/paged_kv.cu
)

target_include_directories(
  glm_asr_infer
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_ROOT}/include
)

target_compile_options(
  glm_asr_infer
  PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-O2 -Wall>
    $<$<COMPILE_LANGUAGE:CUDA>:-O2 -arch=${GLMASR_CUDA_ARCH} --compiler-options=-Wall,-Wno-unused-function>
)

target_link_directories(glm_asr_infer PRIVATE ${CUDA_ROOT}/lib64)
target_link_libraries(
  glm_asr_infer
  PRIVATE
    cublas
    cublasLt
    cufft
    cudart
    cuda
    m
    dl
)

set_target_properties(
  glm_asr_infer
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    BUILD_RPATH "${GLMASR_EXTRA_RPATH}"
    INSTALL_RPATH "${GLMASR_EXTRA_RPATH}"
)

add_custom_target(
  cutedsl_fa2_bridge
  COMMAND
    ${CMAKE_COMMAND} -E env
      CUDA_ROOT=${CUDA_ROOT}
      GLMASR_CUDA_ARCH=${GLMASR_CUDA_ARCH}
      PYTHON=${GLMASR_PYTHON_EXECUTABLE}
      bash ${CMAKE_CURRENT_SOURCE_DIR}/cutedsl_fa/build_bridge.sh
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Building optional CuteDSL FA2 bridge: cutedsl_fa/fa2_bridge.so"
)
