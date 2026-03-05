# Build ITS neighbor-search MEX files (nn_prepare, nn_search, range_search).
# Supports Linux and macOS; run from project root. Requires: octave, mkoctfile (liboctave-dev).

MEX_DIR := matlab/mex
# Use absolute include paths so headers are found on all platforms (macOS clang resolves -I differently)
MEX_ABS := $(abspath $(MEX_DIR))
INC := -I$(MEX_ABS)/tstool -I$(MEX_ABS)/tstool/NN -I$(MEX_ABS)/tstool/mextools
MKOCTFILE ?= mkoctfile

# Same flags for Linux and macOS (README); -D_LIBCPP_... for modern libc++ on macOS.
MEX_FLAGS := --mex -DMATLAB_MEX_FILE -D_LIBCPP_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION -O3

SOURCES := $(MEX_DIR)/tstool/NN/nn_prepare.cpp $(MEX_DIR)/tstool/NN/nn_search.cpp $(MEX_DIR)/tstool/NN/range_search.cpp

.PHONY: mex docs clean help
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
	@echo "Targets: mex (build MEX files), docs (build Sphinx HTML), clean, help. Supported: Linux, macOS."
