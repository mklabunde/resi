#!/bin/bash

sessionName="run-sst2-aug"
tmux kill-session -t $sessionName
tmux new -s $sessionName -d
tmux send-keys -t $sessionName "source .venv/bin/activate" Enter
tmux send-keys -t $sessionName "clear" Enter
tmux send-keys -t $sessionName "python repsim/run.py -c repsim/configs/nlp_augmentation_sst2.yaml" Enter
