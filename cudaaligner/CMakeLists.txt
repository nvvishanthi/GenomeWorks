#
# Copyright 2019-2020 NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#



project(cudaaligner)

# Project specific NVCC flags
if((CUDA_VERSION_MAJOR GREATER 10) OR (CUDA_VERSION_MAJOR EQUAL 10 AND CUDA_VERSION_MINOR GREATER 0))
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14 --expt-relaxed-constexpr")
else()
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14")
endif()
message(STATUS "nvcc flags for cudaaligner: ${CUDA_NVCC_FLAGS}")

get_property(gw_library_type GLOBAL PROPERTY gw_library_type)
cuda_add_library(cudaaligner ${gw_library_type}
    src/cudaaligner.cpp
    src/aligner.cpp
    src/alignment.cpp
    src/alignment_impl.cpp
    src/aligner_global.cpp
    src/aligner_global_ukkonen.cpp
    src/aligner_global_myers.cpp
    src/aligner_global_myers_banded.cpp
    src/aligner_global_hirschberg_myers.cpp
    src/needleman_wunsch_cpu.cpp
    src/ukkonen_cpu.cpp
    src/ukkonen_gpu.cu
    src/myers_gpu.cu
    src/hirschberg_myers_gpu.cu
    )

add_library(cudaaligner_internal INTERFACE)
target_include_directories(cudaaligner_internal INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(cudaaligner gwbase cub)

target_compile_options(cudaaligner PRIVATE -Werror -Wall -Wextra)
if (gw_optimize_for_native_cpu)
    target_compile_options(cudapoa PRIVATE -march=native)
endif()

target_include_directories(cudaaligner
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_compile_options(cudaaligner PRIVATE -Werror)

add_doxygen_source_dir(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add tests folder
add_subdirectory(tests)
add_subdirectory(benchmarks)
add_subdirectory(samples)

install(TARGETS cudaaligner
    COMPONENT logging
    EXPORT cudaaligner
    DESTINATION lib
    INCLUDES DESTINATION include
)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT cudaaligner DESTINATION cmake)

# Add auto formatting.
gw_enable_auto_formatting("${CMAKE_CURRENT_SOURCE_DIR}")
