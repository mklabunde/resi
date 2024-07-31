#!/bin/bash

sessionName="run-corr-mnli-aug"
tmux kill-session -t $sessionName
tmux new -s $sessionName -d
tmux send-keys -t $sessionName "source .venv/bin/activate" Enter
tmux send-keys -t $sessionName "clear" Enter
tmux send-keys -t $sessionName "CUDA_VISIBLE_DEVICES=2 python repsim/run.py -c repsim/configs/correlation_nlp_augmentation_mnli.yaml" Enter
