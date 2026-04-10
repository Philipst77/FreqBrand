#!/bin/bash
#SBATCH --job-name=freqbrand_pop_ablation
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
export MPLCONFIGDIR=/scratch/ygoonati/freqbrand/.cache/matplotlib

cd /scratch/ygoonati/freqbrand
mkdir -p logs results/ablation_population_size

echo "FreqBrand: Population size ablation"
echo "Job ID: $SLURM_JOB_ID  |  Node: $SLURM_NODELIST"

python scripts/ablation_population_size.py \
    --spec_root  results/phase3_spectra/spectra \
    --model_path results/phase3_detection/resnet18_classifier.pt \
    --out_dir    results/ablation_population_size

echo "Population size ablation complete."
echo "Results: results/ablation_population_size/"
