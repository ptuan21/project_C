# Makefile cho mlcpp — chạy được ngay trên macOS (Apple Silicon) không cần cmake.
CXX      := clang++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Iinclude
LDFLAGS  :=

# Trên macOS, dùng Accelerate framework cho phép nhân ma trận tối ưu.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  CXXFLAGS += -DMLCPP_USE_ACCELERATE -DACCELERATE_NEW_LAPACK
  LDFLAGS  += -framework Accelerate
endif

BUILD        := build
SRC          := $(shell find src -name '*.cpp')
OBJ          := $(SRC:.cpp=.o)
TEST_SRC     := $(wildcard tests/*.cpp)
EXAMPLES     := $(wildcard examples/*.cpp)
EXAMPLE_BINS := $(patsubst examples/%.cpp,$(BUILD)/%,$(EXAMPLES))

.PHONY: all demo test clean portable

all: demo test

# --- build GPT inference portable: thuần C++, KHÔNG Accelerate (chạy mọi bo mạch) ---
PORTABLE_SRC := $(shell find src -name '*.cpp')
portable: | $(BUILD)
	$(CXX) -std=c++17 -O2 -Iinclude $(PORTABLE_SRC) examples/gpt_demo.cpp \
	  -o $(BUILD)/gpt_portable
	@echo ">> Đã build portable (không Accelerate): $(BUILD)/gpt_portable"

# --- demo: build mọi file trong examples/ ---
demo: $(EXAMPLE_BINS)
	@echo ">> Đã build: $(EXAMPLE_BINS)"

$(BUILD)/%: examples/%.cpp $(OBJ) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(OBJ) $< -o $@ $(LDFLAGS)

# --- test ---
test: $(BUILD)/run_tests
	@./$(BUILD)/run_tests

$(BUILD)/run_tests: $(OBJ) $(TEST_SRC) | $(BUILD)
	$(CXX) $(CXXFLAGS) -Itests $(OBJ) $(TEST_SRC) -o $@ $(LDFLAGS)

# --- quy tắc chung ---
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD):
	@mkdir -p $(BUILD)

clean:
	find src -name '*.o' -delete
	rm -rf $(BUILD)
