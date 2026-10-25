#!/bin/bash
# Confirm the explicit solver is stable across the production prior (m_phi in [0.6,1.4] x amp
# in [1e-4,1e-1]) at the PRODUCTION grid N=257, before submitting the regular chain.
#SBATCH -A <account>
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:20:00
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -c 32
#SBATCH -J n257_check
#SBATCH -o logs/n257_check_%j.out
#SBATCH -e logs/n257_check_%j.out
# Set CONDA_ENV, NEUTT_ROOT, LINX_TRANSPORT_ROOT and ACT_BBN_ROOT before submitting.

module load conda
conda activate "$CONDA_ENV"
export OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPICH_GPU_SUPPORT_ENABLED=1
export PYTHONUNBUFFERED=1
export JAX_COMPILATION_CACHE_DIR=$NEUTT_ROOT/.jaxcache
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0.0
export PYTHONPATH=$NEUTT_ROOT:$LINX_TRANSPORT_ROOT:$NEUTT_ROOT/mcmc
export SWEEP_N=257
cd $NEUTT_ROOT
echo "############ N=257 production-prior corner stability ############"
python mcmc/sweep_stability.py explicit 2>&1 | grep --line-buffered -vE "cuda_executor|NVML|GPU interconnect"
echo "############ DONE (exit ${PIPESTATUS[0]}) ############"
