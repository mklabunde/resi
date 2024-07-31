#!/bin/bash

sessionNameCompare="run"
tmux new -s $sessionNameCompare -d
tmux send-keys -t $sessionNameCompare "source .venv/bin/activate" Enter
tmux send-keys -t $sessionNameCompare "clear" Enter

sessionNameTrain="train"
tmux new -s $sessionNameTrain -d
tmux send-keys -t $sessionNameTrain "source .venv/bin/activate" Enter
tmux send-keys -t $sessionNameTrain "clear" Enter
