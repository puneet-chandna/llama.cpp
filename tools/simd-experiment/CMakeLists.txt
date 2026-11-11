# simd-experiment: microbenchmark and differential tester for the q8_K
# activation quantizer. Both tools call ggml internals directly, so they are
# only built when the CPU backend is compiled in-process (see tools/CMakeLists.txt).

set(TARGET simd-bench)
add_executable(${TARGET} simd-bench.cpp corpus.h)
target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src ${PROJECT_SOURCE_DIR}/ggml/src/ggml-cpu)
target_link_libraries(${TARGET} PRIVATE ggml)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()

set(TARGET simd-diff)
add_executable(${TARGET} simd-diff.cpp corpus.h)
target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src ${PROJECT_SOURCE_DIR}/ggml/src/ggml-cpu)
target_link_libraries(${TARGET} PRIVATE ggml)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
