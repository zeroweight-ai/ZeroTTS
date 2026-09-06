cmake_minimum_required(VERSION 3.16)
project(zerotts-ggml CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ggml's own CPU backend is all this needs — the point of the port is quantized
# matvec throughput, not an accelerator.
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

if (EMSCRIPTEN)
    # WASM SIMD is the entire reason this is worth doing in a browser; without
    # it ggml falls back to scalar kernels and loses to onnxruntime-web.
    set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
    add_compile_options(-msimd128 -pthread -O3)
endif()

add_subdirectory(vendor-ggml)

add_library(zerotts STATIC src/zerotts.cpp)
target_include_directories(zerotts PUBLIC include)
target_link_libraries(zerotts PUBLIC ggml)

if (EMSCRIPTEN)
    add_executable(zerotts-wasm src/wasm.cpp)
    target_link_libraries(zerotts-wasm PRIVATE zerotts)
    target_link_options(zerotts-wasm PRIVATE
        -sMODULARIZE=1
        -sEXPORT_ES6=1
        -sEXPORT_NAME=createZeroTTS
        -sALLOW_MEMORY_GROWTH=1
        -sMAXIMUM_MEMORY=4GB
        -sSTACK_SIZE=1048576
        -sEXPORTED_RUNTIME_METHODS=['HEAPU8','HEAPF32','HEAP32','ccall','cwrap']
        -sPTHREAD_POOL_SIZE=8
        -sENVIRONMENT=web,worker,node
        -pthread
        -O3
    )
else()
    add_executable(zerotts-bench test/bench.cpp)
    target_link_libraries(zerotts-bench PRIVATE zerotts)

    add_executable(zerotts-quality test/quant_quality.cpp)
    target_link_libraries(zerotts-quality PRIVATE zerotts)
endif()
