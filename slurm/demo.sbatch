#!/bin/bash
#SBATCH --job-name=test_job
#SBATCH --output=/shared_work/markhsp/output_%j.log
#SBATCH --error=/shared_work/markhsp/error_%j.log
#SBATCH --partition=gb10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --time=00:10:00

echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU(s) allocated: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

# Count for 5 minutes (300 seconds)
for i in $(seq 1 300); do
    echo "Second $i / 300"
    sleep 1
done

echo "Finished at $(date)"
