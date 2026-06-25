# Makefile — dự án 3 mảng: lib/ (thư viện mlcpp), apps/ (demo GPT), patent/ (hệ thống patent).
# Chạy ngay trên macOS (Apple Silicon) không cần cmake.
CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -fopenmp -Ilib/include
LDFLAGS  := -fopenmp

# Trên macOS, dùng Accelerate framework cho phép nhân ma trận tối ưu.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  CXXFLAGS += -DMLCPP_USE_ACCELERATE -DACCELERATE_NEW_LAPACK
  LDFLAGS  += -framework Accelerate
endif

BUILD     := build
LIB_SRC   := $(shell find lib/src -name '*.cpp')
LIB_OBJ   := $(LIB_SRC:.cpp=.o)
TEST_SRC  := $(wildcard tests/*.cpp)
APPS      := $(wildcard apps/*.cpp)
APP_BINS  := $(patsubst apps/%.cpp,$(BUILD)/%,$(APPS))
PATENT_SRC := $(shell find patent/src -name '*.cpp' 2>/dev/null)

.PHONY: all apps test clean portable patent patent-test

all: apps test

# --- demo GPT: build mọi file trong apps/ ---
apps: $(APP_BINS)
	@echo ">> Đã build: $(APP_BINS)"

$(BUILD)/%: apps/%.cpp $(LIB_OBJ) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(LIB_OBJ) $< -o $@ $(LDFLAGS)

# --- hệ thống patent (CLI) ---
patent: patent/cli/patent.cpp $(LIB_OBJ) $(PATENT_SRC) | $(BUILD)
	$(CXX) $(CXXFLAGS) -Ipatent/include $(LIB_OBJ) $(PATENT_SRC) patent/cli/patent.cpp \
	  -o $(BUILD)/patent $(LDFLAGS)
	@echo ">> Đã build: $(BUILD)/patent"

# --- test patent (BM25 + risk) ---
patent-test: | $(BUILD)
	$(CXX) $(CXXFLAGS) -Ipatent/include -Itests \
	  patent/src/bm25.cpp patent/src/risk.cpp lib/src/data/tokenizer.cpp \
	  patent/tests/test_bm25.cpp patent/tests/test_risk.cpp patent/tests/main.cpp \
	  -o $(BUILD)/patent_test $(LDFLAGS)
	@./$(BUILD)/patent_test

# --- test ---
test: $(BUILD)/run_tests
	@./$(BUILD)/run_tests

$(BUILD)/run_tests: $(LIB_OBJ) $(TEST_SRC) | $(BUILD)
	$(CXX) $(CXXFLAGS) -Itests $(LIB_OBJ) $(TEST_SRC) -o $@ $(LDFLAGS)

# --- GPT inference portable: thuần C++, KHÔNG Accelerate (bo mạch yếu) ---
portable: | $(BUILD)
	$(CXX) -std=c++17 -O2 -Ilib/include $(LIB_SRC) apps/gpt_demo.cpp -o $(BUILD)/gpt_portable
	@echo ">> Đã build portable (không Accelerate): $(BUILD)/gpt_portable"

# --- quy tắc chung ---
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD):
	@mkdir -p $(BUILD)

clean:
	find lib/src patent -name '*.o' -delete 2>/dev/null || true
	rm -rf $(BUILD)
