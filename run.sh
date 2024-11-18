#!/bin/bash

EXP_NAME="ViTB_16_CC3M"
TRAIN_DATA=/raid/sedigheh.eslami/datasets/cc3m/Train_GCC-training_output.tsv
MODEL_NAME=ViT-B/16
BS=512
LR=1e-3
EPOCHS=30
LOGS=/raid/sedigheh.eslami/outputs/CyCLIP

python -m src.main --name $EXP_NAME --train_data $TRAIN_DATA --delimiter "\t" --image_key "filepath" --caption_key "title" --cylambda1 0.25 --cylambda2 0.25 --model_name $MODEL_NAME --batch_size $BS --epochs $EPOCHS --lr $LR --wandb --logs $LOGS --num_workers 2 --distributed 