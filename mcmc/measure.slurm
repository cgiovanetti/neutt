#!/bin/bash
# Authoritative per-point cost + no-recompile + physics on the real make_coupled_bbn pipeline,
# at N=101 (fast) and N=257 (production candidate). 1 GPU, debug.
#SBATCH -A <account>
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:29:00
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -c 32
#SBATCH -J sd_measure
#SBATCH -o logs/sd_measure_%j.out
#SBATCH -e logs/sd_measure_%j.out
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
cd $NEUTT_ROOT

for N in 101 257; do
  echo "############ N=$N ############"
  timeout 800 python mcmc/measure_mcmc_point.py $N 2>&1 | grep --line-buffered -vE "cuda_executor|NVML|GPU interconnect"
  echo "   (exit ${PIPESTATUS[0]})"
done
echo "############ DONE ############"
