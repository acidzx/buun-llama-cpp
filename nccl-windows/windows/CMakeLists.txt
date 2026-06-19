cmake_minimum_required(VERSION 3.20)
project(nccl CUDA)

add_library(nccl.13.3 SHARED
    ../src/all_gather.cu
    ../src/all_reduce.cu
    ../src/broadcast.cu
    ../src/core.cu
    ../src/reduce.cu
    ../src/reduce_scatter.cu
    ../src/libwrap.cu
)

target_compile_definitions(nccl.13.3 PRIVATE COMPILING_NCCL)

target_include_directories(nccl.13.3 PUBLIC ../include)

set_target_properties(nccl.13.3 PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/nccl-output"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/nccl-output"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/nccl-output"
    OUTPUT_NAME "nccl64_134.13.3"
    CUDA_ARCHITECTURES "86;89"
)
