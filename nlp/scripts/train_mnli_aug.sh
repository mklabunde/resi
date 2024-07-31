#!/bin/bash

device_map=(0 0 1 2 3 4)

for seed in {1..4}; do
    sessionNameTrain="train-mnli-aug-$seed"
    tmux kill-session -t $sessionNameTrain
    tmux new -s $sessionNameTrain -d
    tmux send-keys -t $sessionNameTrain "source .venv/bin/activate" Enter
    tmux send-keys -t $sessionNameTrain "clear" Enter
    tmux send-keys -t $sessionNameTrain "CUDA_VISIBLE_DEVICES=${device_map[$seed]} python nlp/bert_finetune.py model.seed=$seed dataset.finetuning.trainer.args.seed=$seed dataset=mnli_eda_025,mnli_eda_05,mnli_eda_075,mnli_eda_10 -m" Enter
    sleep 1
done
