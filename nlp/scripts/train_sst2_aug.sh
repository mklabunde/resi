#!/bin/bash

sessionNameTrain="train-sst2-aug"
tmux new -s $sessionNameTrain -d
tmux send-keys -t $sessionNameTrain "source .venv/bin/activate" Enter
tmux send-keys -t $sessionNameTrain "clear" Enter

# Train augmented models
for seed in {0..9}; do
    tmux send-keys -t $sessionNameTrain "python nlp/bert_finetune.py model.seed=$seed dataset=sst2 dataset.finetuning.trainer.args.seed=$seed augmentation=eda augmentation.recipe.pct_words_to_swap=1,0.75,0.5,0.25 -m" Enter
done
