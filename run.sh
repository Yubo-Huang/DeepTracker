#!/bin/bash

# Run the LSTM training script
# nohup ./run.sh > lstm.log 2>&1 & echo $!
rm -r logs_multitask_cls/*
rm -r checkpoints_multitask_cls/*
python LSTM-Reg_train.py
# python train.py