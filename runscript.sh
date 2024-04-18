#!/bin/bash
#SBATCH -c 1
#SBATCH -t 0-14:00
#SBATCH --partition=gpu,seas_gpu
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --mem=82000
#SBATCH -o ./run_outputs/out/myoutput_%j.out
#SBATCH -e ./run_outputs/errors/myerrors_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=vladimir.petrov.239@gmail.com 


module load python/3.10.9-fasrc01

conda activate RL-transformer

python main.py run_config.yaml