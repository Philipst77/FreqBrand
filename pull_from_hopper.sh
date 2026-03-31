#!/bin/bash
rsync -avz \
  --exclude='.cache/' \
  --exclude='checkpoints/' \
  --exclude='data/poisoned_datasets/' \
  --exclude='data/clean_finetune_data/' \
  --exclude='silent-branding-attack/' \
  --exclude='results/phase1_sanity/spectra/' \
  --exclude='results/phase3_generation/' \
  ygoonati@hopper.orc.gmu.edu:/scratch/ygoonati/freqbrand/ \
  /Users/ygoonati/freqbrand/
