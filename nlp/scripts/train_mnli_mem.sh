#!/bin/bash

device_map=(0 1 2 3 4)

for seed in {0..4}; do
    if [ $seed -eq 1 ]; then
        continue
    fi
    sessionNameTrain="train-mnli-mem-$seed"
    tmux kill-session -t $sessionNameTrain
    tmux new -s $sessionNameTrain -d
    tmux send-keys -t $sessionNameTrain "source .venv/bin/activate" Enter
    tmux send-keys -t $sessionNameTrain "clear" Enter
    tmux send-keys -t $sessionNameTrain "CUDA_VISIBLE_DEVICES=${device_map[$seed]} python nlp/bert_finetune.py dataset.finetuning.trainer.args.num_train_epochs=10 dataset.finetuning.trainer.args.seed=$seed dataset=mnli memorization_rate=0.25,0.5,0.75,1.0 model.seed=$seed -m" Enter
    sleep 1
done
