#!/bin/bash
sessionNameTrain="train-sst2-memo"

tmux kill-session -t $sessionNameTrain

tmux new -s $sessionNameTrain -d
tmux send-keys -t $sessionNameTrain "source .venv/bin/activate" Enter
tmux send-keys -t $sessionNameTrain "clear" Enter

# Train memorizing models
for seed in {0..4}; do
    tmux send-keys -t $sessionNameTrain "python nlp/bert_finetune.py model.seed=$seed dataset=sst2 dataset.finetuning.trainer.args.seed=$seed memorization_rate=1.0,0.75,0.5,0.25 -m" Enter
done
