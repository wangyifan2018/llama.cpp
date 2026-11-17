add_executable(llama-paraformer main.cpp paraformer.cpp core/gguf_loader.cpp core/kaldi_fbank.cpp)
target_compile_features(llama-paraformer PRIVATE cxx_std_17)
target_include_directories(llama-paraformer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(llama-paraformer SYSTEM PRIVATE ${PROJECT_SOURCE_DIR}/vendor)
target_link_libraries(llama-paraformer PRIVATE ggml Threads::Threads ${CMAKE_DL_LIBS})
install(TARGETS llama-paraformer RUNTIME)

add_executable(llama-paraformer-server server.cpp paraformer.cpp core/gguf_loader.cpp core/kaldi_fbank.cpp)
target_compile_features(llama-paraformer-server PRIVATE cxx_std_17)
target_include_directories(llama-paraformer-server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(llama-paraformer-server SYSTEM PRIVATE ${PROJECT_SOURCE_DIR}/vendor)
target_link_libraries(llama-paraformer-server PRIVATE ggml Threads::Threads ${CMAKE_DL_LIBS})
install(TARGETS llama-paraformer-server RUNTIME)

# Offline re-quantization helper (bench tooling, not installed).
add_executable(llama-paraformer-requant tools/requant.cpp)
target_compile_features(llama-paraformer-requant PRIVATE cxx_std_17)
target_link_libraries(llama-paraformer-requant PRIVATE ggml)
