# CMake config file for custom MyCaffe/NCCL build with CUDA 13.3
# This provides the NCCL::NCCL target for llama.cpp CMake

if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/nccl64_134.13.3.lib")
    add_library(NCCL::NCCL INTERFACE IMPORTED)
    set_target_properties(NCCL::NCCL PROPERTIES
        INTERFACE_LINK_LIBRARIES "${CMAKE_CURRENT_LIST_DIR}/nccl64_134.13.3.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/.."
    )
endif()
