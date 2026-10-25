#!/bin/bash
# Short BBN MCMC for the OOE scalar->nu decay. 4 chains, 1 GPU each.
# Submit to regular (production) once the harness is debug-validated:
#   sbatch mcmc/scalar_decay_mcmc.slurm
#SBATCH -A <account>
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 10:00:00
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH -c 32
#SBATCH -J scalar_decay_mcmc
#SBATCH -o logs/scalar_decay_mcmc_%j.out
#SBATCH -e logs/scalar_decay_mcmc_%j.out
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

# pre-warm the persistent compile cache (1 rank) so the 4-rank launch loads instead of contending
echo "### pre-warming compile cache (1 rank) ###"
srun --ntasks=1 --gpus=1 --cpu-bind=cores python mcmc/warm_cache.py 2>&1 | grep -aE "cache warmed|Error|Traceback"

# one GPU per MPI rank (one cobaya chain per GPU)
echo "### launching 4-chain MCMC ###"
srun --ntasks=4 --gpu-bind=none --cpu-bind=cores bash -c '
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  python mcmc/run_mcmc.py mcmc/scalar_decay_bbn.yaml
'
