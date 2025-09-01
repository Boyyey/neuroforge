CC = gcc
NVCC = nvcc
CFLAGS = -Wall -Wextra -O3 -march=native -fopenmp
CUDA_FLAGS = -arch=sm_70 -O3 -Xcompiler "-fopenmp"
LDFLAGS = -lm -fopenmp
CUDA_LDFLAGS = -lcudart -lcublas -lcurand

# BLAS configuration (uncomment based on your system)
# BLAS_LDFLAGS = -lopenblas
# BLAS_LDFLAGS = -lblas
BLAS_LDFLAGS =

SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

SOURCES = $(wildcard $(SRC_DIR)/*.c) \
          $(wildcard $(SRC_DIR)/layers/*.c) \
          $(wildcard $(SRC_DIR)/optimizers/*.c) \
          $(wildcard $(SRC_DIR)/activations/*.c)

CUDA_SOURCES = $(wildcard $(SRC_DIR)/cuda/*.cu)
CUDA_OBJECTS = $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CUDA_SOURCES))

OBJECTS = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SOURCES))
EXAMPLES = $(wildcard examples/*.c)
TESTS = $(wildcard tests/*.c)

# Determine if CUDA is available
HAVE_CUDA = $(shell which $(NVCC) >/dev/null 2>&1 && echo 1 || echo 0)

ifeq ($(HAVE_CUDA), 1)
	CFLAGS += -DUSE_CUDA
	LDFLAGS += $(CUDA_LDFLAGS)
	OBJECTS += $(CUDA_OBJECTS)
endif

.PHONY: all clean clear examples tests

all: $(BIN_DIR)/libnn.a

$(BIN_DIR)/libnn.a: $(OBJECTS)
	@mkdir -p $(BIN_DIR)
	ar rcs $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(CUDA_FLAGS) -c $< -o $@

examples: $(BIN_DIR)/libnn.a
	@for example in $(EXAMPLES); do \
		name=$$(basename $${example%.c}); \
		echo "Building example $$name..."; \
		$(CC) $(CFLAGS) $$example -o $(BIN_DIR)/$$name $(BIN_DIR)/libnn.a $(LDFLAGS) $(BLAS_LDFLAGS); \
	done

tests: $(BIN_DIR)/libnn.a
	@for test in $(TESTS); do \
		name=$$(basename $${test%.c}); \
		echo "Building test $$name..."; \
		$(CC) $(CFLAGS) $$test -o $(BIN_DIR)/$$name $(BIN_DIR)/libnn.a $(LDFLAGS) $(BLAS_LDFLAGS); \
		echo "Running $$name..."; \
		./$(BIN_DIR)/$$name; \
	done

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

clear: clean
	@echo "Cleaned all build artifacts"