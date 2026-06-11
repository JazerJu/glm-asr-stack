NVCC     = /usr/local/cuda-12.8/bin/nvcc
CUDA_INC = /usr/local/cuda-12.8/include
CC       = gcc
CXX      = g++
CFLAGS   = -O2 -Wall -Iinclude -I$(CUDA_INC)
CXXFLAGS = -O2 -Wall -std=c++17 -Iinclude -I$(CUDA_INC)
NVFLAGS  = -O2 -std=c++17 -arch=sm_120 -Iinclude --compiler-options "-Wall -Wno-unused-function"
LDFLAGS  = -L/usr/local/cuda-12.8/lib64 -lcublas -lcublasLt -lcufft -lcudart -lcuda -lm

BUILD    = build
TARGET   = force_aligner

C_SRC    = src/safetensors.cpp src/tokenizer.cpp src/audio_io.cpp src/model.cpp \
           src/text_processor.cpp src/timestamp.cpp src/aligner.cpp src/main.cpp
CU_SRC   = cuda/kernels.cu cuda/mel.cu cuda/audio_encoder.cu cuda/text_decoder.cu

C_OBJ    = $(patsubst src/%.cpp,$(BUILD)/%.o,$(C_SRC))
CU_OBJ   = $(patsubst cuda/%.cu,$(BUILD)/%.cu.o,$(CU_SRC))

MODEL_DIR = /data/.cache/huggingface/hub/models--Qwen--Qwen3-ForcedAligner-0.6B/snapshots/c7cbfc2048c462b0d63a45797104fc9db3ad62b7
TEST_AUDIO = /data/fwsr/glm-asr/GLM-ASR/cut.wav

all: $(BUILD) $(TARGET)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/%.o: src/%.cpp include/*.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/%.cu.o: cuda/%.cu include/*.h cuda/kernels.cuh | $(BUILD)
	$(NVCC) $(NVFLAGS) -c $< -o $@

$(TARGET): $(C_OBJ) $(CU_OBJ)
	$(NVCC) $(NVFLAGS) $^ -o $@ $(LDFLAGS)

run: $(TARGET)
	./$(TARGET) \
		--model $(MODEL_DIR) \
		--audio $(TEST_AUDIO) \
		--text "今天天气很好，我们出去玩吧" \
		--lang Chinese

clean:
	rm -rf $(BUILD) $(TARGET)

.PHONY: all clean run
