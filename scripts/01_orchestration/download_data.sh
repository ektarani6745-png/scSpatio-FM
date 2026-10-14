#!/bin/bash
set -e
echo "Initializing scSpatio-FM Step 1: Downloading 10x Genomics Visium Data via Nextflow..."
nextflow run main.nf
echo "Orchestration complete. Raw data staged to ../../data/raw/"
