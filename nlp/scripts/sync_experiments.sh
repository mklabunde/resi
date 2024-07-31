#!/bin/bash
sessionName="minio-down"

tmux kill-session -t $sessionName

tmux new -s $sessionName -d

resultPath="/root/similaritybench/experiments"
tmux send-keys -t $sessionName "mc mirror innkube/datasets/simbench $resultPath" Enter
