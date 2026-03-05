# Build ITS neighbor-search MEX files (nn_prepare, nn_search, range_search).
# Supports Linux and macOS; run from project root. Requires: octave, mkoctfile (liboctave-dev).

MEX_DIR := matlab/mex
# Include paths relative to MEX_DIR (we cd there before mkoctfile)
INC := -Itstool -Itstool/NN -Itstool/mextools
MKOCTFILE ?= mkoctfile

# Same flags for Linux and macOS (README); -D_LIBCPP_... for modern libc++ on macOS.
MEX_FLAGS := --mex -DMATLAB_MEX_FILE -D_LIBCPP_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION -O3

SOURCES := $(MEX_DIR)/tstool/NN/nn_prepare.cpp $(MEX_DIR)/tstool/NN/nn_search.cpp $(MEX_DIR)/tstool/NN/range_search.cpp

.PHONY: mex docs clean help
mex: $(SOURCES)
	cd $(MEX_DIR) && $(MKOCTFILE) $(MEX_FLAGS) $(INC) tstool/NN/nn_prepare.cpp -o nn_prepare
	cd $(MEX_DIR) && $(MKOCTFILE) $(MEX_FLAGS) $(INC) tstool/NN/nn_search.cpp -o nn_search
	cd $(MEX_DIR) && $(MKOCTFILE) $(MEX_FLAGS) $(INC) tstool/NN/range_search.cpp -o range_search

docs:
	cd docs && $(MAKE) html

clean:
	cd $(MEX_DIR) && rm -f nn_prepare nn_search range_search *.o *.mex*
	cd docs && $(MAKE) clean

help:
	@echo "Targets: mex (build MEX files), docs (build Sphinx HTML), clean, help. Supported: Linux, macOS."
