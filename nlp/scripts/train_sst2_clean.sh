#!/bin/bash

sessionNameTrain="train-sst2-clean"
tmux new -s $sessionNameTrain -d
tmux send-keys -t $sessionNameTrain "source .venv/bin/activate" Enter
tmux send-keys -t $sessionNameTrain "clear" Enter

# Train clean models
for seed in {0..9}; do
    tmux send-keys -t $sessionNameTrain "python nlp/bert_finetune.py model.seed=$seed dataset=sst2 dataset.finetuning.trainer.args.seed=$seed" Enter
done
