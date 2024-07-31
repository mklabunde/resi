#!/bin/bash
sessionName="minio"

tmux kill-session -t $sessionName

tmux new -s $sessionName -d

resultPath="/root/similaritybench/experiments"
tmux send-keys -t $sessionName "mc mirror $resultPath innkube/datasets/simbench" Enter
