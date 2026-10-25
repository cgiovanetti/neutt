#!/bin/bash
# MCMC pipeline smoke (1 GPU, debug): (1) no-recompile check across 2 dark params,
# (2) one full cobaya evaluation of the theory+likelihood. Validates the whole MCMC path
# before submitting the production chain to regular.
#SBATCH -A <account>
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:29:00
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -c 32
#SBATCH -J sd_smoke
#SBATCH -o logs/sd_smoke_%j.out
#SBATCH -e logs/sd_smoke_%j.out
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
export PYTHONPATH=$NEUTT_ROOT:$LINX_TRANSPORT_ROOT:$ACT_BBN_ROOT:$NEUTT_ROOT/mcmc

echo "############ no-recompile check (2 dark params reuse one compiled solve) ############"
cd $LINX_TRANSPORT_ROOT
timeout 700 python -m transport_integration.verify_no_recompile 2>&1 | grep --line-buffered -vE "cuda_executor|NVML|GPU interconnect"
echo "   (exit ${PIPESTATUS[0]})"

echo "############ cobaya evaluate (one full theory+likelihood eval, N=101) ############"
cd $NEUTT_ROOT
timeout 700 python mcmc/run_mcmc.py mcmc/scalar_decay_test.yaml 2>&1 | grep --line-buffered -vE "cuda_executor|NVML|GPU interconnect"
echo "   (exit ${PIPESTATUS[0]})"
echo "############ DONE ############"
