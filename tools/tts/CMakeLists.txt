set(TARGET llama-tts)
add_executable(${TARGET} tts.cpp)
target_link_libraries(${TARGET} PRIVATE llama llama-common mtmd ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()

# vocbench: code2wav alone, for timing/backends A/B
set(TARGET llama-vocbench)
add_executable(${TARGET} vocbench.cpp)
target_link_libraries(${TARGET} PRIVATE llama llama-common mtmd ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# mmabench: batching vs weight streaming for the code predictor's matmul shapes
set(TARGET llama-mmabench)
add_executable(${TARGET} mmabench.cpp)
target_link_libraries(${TARGET} PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
