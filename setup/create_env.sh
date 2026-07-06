#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

#SBATCH --job-name=env_creation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=01:00:00

# Exit immediately if a command exits with a non-zero status
set -e

# Start timer
start_time=$(date +%s)

# Get the current date
current_date=$(date +%y%m%d)

# Create environment name with the current date
env_prefix=lingua_$current_date

# Create the conda environment

# Fall back to the active conda base if CONDA_ROOT is not exported (e.g. on a login node).
CONDA_ROOT="${CONDA_ROOT:-$(conda info --base 2>/dev/null)}"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda create -n $env_prefix python=3.11 -y -c anaconda
conda activate $env_prefix

echo "Currently in env $(which python)"

# Install packages.
# IMPORTANT: SDM's Triton kernels use bf16 atomic_add, which requires triton >= 3.4.
# That ships with torch 2.8 — torch 2.7 bundles triton 3.3 and the kernels fail to compile.
pip install torch==2.8.0 xformers
pip install ninja
pip install --requirement requirements.txt

# CUDA toolkit (nvcc + cudart + libs) matching torch's CUDA build, for JIT kernel
# compilation via torch.utils.cpp_extension.load_inline. Requires an SM80+ GPU at runtime.
#
# NOTE: cuda-nvcc pulls in a conda cross-compiler (gxx_linux-64) as a dependency. On some
# channel combinations an old build's post-link script emits a benign nonzero warning
# ("ERROR: This cross-compiler package contains no program ...-c++"). We compile the
# kernels with the system g++ (configured in the activation hook below), so this is
# harmless — tolerate the warning but hard-verify that nvcc actually landed.
if ! conda install -y -c nvidia cuda-toolkit="$(python -c 'import torch; print(torch.version.cuda)')"; then
    echo "WARN: 'conda install cuda-toolkit' returned nonzero (usually a benign cross-compiler post-link warning); verifying nvcc..."
fi
command -v nvcc >/dev/null 2>&1 || { echo "FATAL: nvcc not found after installing cuda-toolkit." >&2; exit 1; }
echo "nvcc OK: $(nvcc --version | tail -1)"

# That old gxx_linux-64 build ships without its g++/c++ driver binaries, so conda's own
# activate/deactivate scripts for it print a scary (harmless) error on every `conda activate`.
# We compile with the system g++ anyway, so point the expected driver names at it — this makes
# conda's compiler activation scripts succeed silently. No-op if a healthy compiler is present.
host_gcc=$(ls "$CONDA_PREFIX"/bin/*-linux-gnu-gcc 2>/dev/null | head -1)
if [ -n "$host_gcc" ] && [ -x /usr/bin/g++ ]; then
    prefix="${host_gcc%-gcc}"
    for drv in c++ g++; do
        [ -e "${prefix}-${drv}" ] || ln -s /usr/bin/g++ "${prefix}-${drv}"
    done
fi

# Make the JIT CUDA-extension build robust: use the system C++ host compiler (the conda
# cross-compiler can be incomplete) and let the linker find libcudart (conda installs it
# under $CONDA_PREFIX/lib, but nvcc searches lib64). Persist via an activation hook.
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/zz_sdm_build_env.sh" <<'EOF'
# Set by sparse-delta-memory setup for robust CUDA-extension JIT builds.
if [ -x /usr/bin/g++ ]; then export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDAHOSTCXX=/usr/bin/g++; fi
export LIBRARY_PATH="${CONDA_PREFIX}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
EOF

# End timer
end_time=$(date +%s)

# Calculate elapsed time in seconds
elapsed_time=$((end_time - start_time))

# Convert elapsed time to minutes
elapsed_minutes=$((elapsed_time / 60))

echo "Environment $env_prefix created and all packages installed successfully in $elapsed_minutes minutes!"


