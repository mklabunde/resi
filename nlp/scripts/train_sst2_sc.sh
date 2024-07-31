#!/bin/bash
sessionNameTrain="train-sst2-sc"

tmux kill-session -t $sessionNameTrain

tmux new -s $sessionNameTrain -d
tmux send-keys -t $sessionNameTrain "source .venv/bin/activate" Enter
tmux send-keys -t $sessionNameTrain "clear" Enter

# Train shortcut models
for seed in {0..9}; do
    # tmux send-keys -t $sessionNameTrain "python nlp/bert_finetune.py model.seed=$seed dataset=sst2 dataset.finetuning.trainer.args.seed=$seed dataset.finetuning.trainer.args.num_train_epochs=3 shortcut_rate=1.0,0.75,0.5,0.25,0 -m" Enter
    # Use different steps, because of class distribution. Step size = (1 - naive_guessing_rate) / 4.
    tmux send-keys -t $sessionNameTrain "python nlp/bert_finetune.py model.seed=$seed dataset=sst2 dataset.finetuning.trainer.args.seed=$seed dataset.finetuning.trainer.args.num_train_epochs=3 shortcut_rate=0.889,0.779,0.668,0.558 -m" Enter
done
