# Build ITS neighbor-search MEX files (nn_prepare, nn_search, range_search).
# Supports Linux and macOS; run from project root. Requires: octave, mkoctfile (liboctave-dev).

UNAME_S := $(shell uname -s)
MEX_DIR := matlab/mex
# Use absolute include paths so headers are found on all platforms (macOS clang resolves -I differently)
MEX_ABS := $(abspath $(MEX_DIR))
INC := -I$(MEX_ABS)/tstool -I$(MEX_ABS)/tstool/NN -I$(MEX_ABS)/tstool/NNSearcher -I$(MEX_ABS)/tstool/mextools
MKOCTFILE ?= mkoctfile

# Same flags for Linux and macOS (README); -D_LIBCPP_... for modern libc++ on macOS.
MEX_FLAGS := --mex -DMATLAB_MEX_FILE -D_LIBCPP_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION -O3

SOURCES := $(MEX_DIR)/tstool/NN/nn_prepare.cpp $(MEX_DIR)/tstool/NN/nn_search.cpp $(MEX_DIR)/tstool/NN/range_search.cpp

.PHONY: mex docs clean help all octave-ready
# Ensure Octave and mkoctfile are available; install if missing (macOS: brew, Linux: apt).
octave-ready:
	@if command -v octave-cli >/dev/null 2>&1 && command -v mkoctfile >/dev/null 2>&1; then \
		echo "Octave and mkoctfile found."; \
	elif [ "$(UNAME_S)" = "Darwin" ]; then \
		echo "Installing Octave via Homebrew..."; \
		brew install octave; \
	elif [ "$(UNAME_S)" = "Linux" ]; then \
		echo "Installing Octave via apt..."; \
		sudo apt-get update && sudo apt-get install -y --no-install-recommends octave octave-dev; \
	else \
		echo "Unsupported OS: $(UNAME_S). Install Octave and mkoctfile manually."; exit 1; \
	fi

# Full pipeline: check/install Octave -> build MEX -> build docs -> run tests.
all: octave-ready mex docs
	pytest tests/ -v

# Run from project root so compiler resolves #include "include.mex" relative to source dir on all platforms
mex: $(SOURCES)
	$(MKOCTFILE) $(MEX_FLAGS) $(INC) $(MEX_DIR)/tstool/NN/nn_prepare.cpp -o $(MEX_DIR)/nn_prepare
	$(MKOCTFILE) $(MEX_FLAGS) $(INC) $(MEX_DIR)/tstool/NN/nn_search.cpp -o $(MEX_DIR)/nn_search
	$(MKOCTFILE) $(MEX_FLAGS) $(INC) $(MEX_DIR)/tstool/NN/range_search.cpp -o $(MEX_DIR)/range_search

docs:
	cd docs && $(MAKE) html

clean:
	cd $(MEX_DIR) && rm -f nn_prepare nn_search range_search *.o *.mex*
	cd docs && $(MAKE) clean

help:
	@echo "Targets: all (octave-ready + mex + docs + pytest), octave-ready (check/install Octave), mex (build MEX files), docs (build Sphinx HTML), clean, help. Supported: Linux, macOS."
