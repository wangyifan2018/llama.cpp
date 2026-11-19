add_executable(llama-qwen3-asr-batch main.cpp)
target_compile_features(llama-qwen3-asr-batch PRIVATE cxx_std_17)
target_link_libraries(llama-qwen3-asr-batch PRIVATE llama llama-common mtmd vendor::nlohmann)
install(TARGETS llama-qwen3-asr-batch RUNTIME)
