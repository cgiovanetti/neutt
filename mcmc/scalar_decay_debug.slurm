#!/bin/bash
# Tiny debug chain: EXACT production launch (4 chains/4 GPUs, MPI + GPU binding, mcmc sampler)
# at max_samples=8, N=257 -> validates the full chain mechanics before the regular submission.
#SBATCH -A <account>
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:29:00
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH -c 32
#SBATCH -J sd_chaindbg
#SBATCH -o logs/sd_chaindbg_%j.out
#SBATCH -e logs/sd_chaindbg_%j.out
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
cd $NEUTT_ROOT
export PYTHONPATH=$NEUTT_ROOT:$LINX_TRANSPORT_ROOT:$ACT_BBN_ROOT:$NEUTT_ROOT/mcmc

rm -rf $NEUTT_ROOT/chains/scalar_decay_debug 2>/dev/null
srun --ntasks=4 --gpu-bind=none --cpu-bind=cores bash -c '
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  python mcmc/run_mcmc.py mcmc/scalar_decay_debug.yaml
' 2>&1 | grep --line-buffered -vE "cuda_executor|NVML|GPU interconnect"
echo "############ DONE (exit ${PIPESTATUS[0]}) ############"
echo "=== chain files ==="; ls -la $NEUTT_ROOT/chains/scalar_decay_debug/ 2>/dev/null
