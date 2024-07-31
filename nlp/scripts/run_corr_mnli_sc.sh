#!/bin/bash

sessionName="run-corr-mnli-sc"
tmux kill-session -t $sessionName
tmux new -s $sessionName -d
tmux send-keys -t $sessionName "source .venv/bin/activate" Enter
tmux send-keys -t $sessionName "clear" Enter
tmux send-keys -t $sessionName "python repsim/run.py -c repsim/configs/correlation_nlp_shortcuts_mnli.yaml" Enter
