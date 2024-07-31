#!/bin/bash

sessionName="run-mnli-aug"
tmux kill-session -t $sessionName
tmux new -s $sessionName -d
tmux send-keys -t $sessionName "source .venv/bin/activate" Enter
tmux send-keys -t $sessionName "clear" Enter
tmux send-keys -t $sessionName "CUDA_VISIBLE_DEVICES=0 python repsim/run.py -c repsim/configs/nlp_augmentation_mnli.yaml" Enter
