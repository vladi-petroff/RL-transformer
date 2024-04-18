# RL-transformer
In-context Reinforcement Learning with GPT transformer

experiments:
run_config.yaml : can specify parameters for a new run (model dimensions, whether to regularize / symmetrize, training settings such as batch size, number of training steps, learning rate and etc)
results will be saved locally (model and losses) and also recorde in wandb as training progresses

runscript.sh : submits the job to the SLURM queue

files:

main.py : model training
ddp_main.py : wrapper to run distributed data parallel training
train_utils.py : loss calculation, training schedulling, and empirical evaluation (via experiment function)
transformer.py : implementation of symmetrized and regularized transformers
env_MAB.py: implementation of bandit class
my_algorithms.py: baseline algorithm (Thompson sampling) and the optimal policy (Gittins index)
