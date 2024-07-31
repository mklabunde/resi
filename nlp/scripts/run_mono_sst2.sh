#!/bin/bash

sessionName="run-mono-sst2"
tmux kill-session -t $sessionName
tmux new -s $sessionName -d
tmux send-keys -t $sessionName "source .venv/bin/activate" Enter
tmux send-keys -t $sessionName "clear" Enter
tmux send-keys -t $sessionName "CUDA_VISIBLE_DEVICES=0 python repsim/run.py -c repsim/configs/monotonicity_nlp_standard_sst2_cls_tok.yaml" Enter
