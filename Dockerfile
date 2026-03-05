# xyz: build ITS MEX files with Octave and run numerical parity tests
FROM python:3.12-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    octave \
    liboctave-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy package and install (README.md required by pyproject.toml)
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir -e . pytest

# Copy matlab (MEX source + ITS), tests, and Makefile
COPY matlab/ matlab/
COPY tests/ tests/
COPY Makefile .

# Build ITS neighbor-search MEX files for this architecture (Linux/macOS via Makefile)
RUN make mex

# Verify MEX and Octave path
RUN octave-cli --eval "addpath('matlab'); addpath('matlab/its'); addpath('matlab/mex'); which nn_prepare; which nn_search; which range_search"

# Run pytest (numerical correctness vs ITS toolbox)
CMD ["pytest", "tests/", "-v"]
