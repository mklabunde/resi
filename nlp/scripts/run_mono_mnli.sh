#!/bin/bash

sessionName="run-mono-mnli"
tmux kill-session -t $sessionName
tmux new -s $sessionName -d
tmux send-keys -t $sessionName "source .venv/bin/activate" Enter
tmux send-keys -t $sessionName "clear" Enter
tmux send-keys -t $sessionName "CUDA_VISIBLE_DEVICES=1 python repsim/run.py -c repsim/configs/monotonicity_nlp_standard_mnli_cls_tok.yaml" Enter
