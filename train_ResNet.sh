#!/usr/bin/env sh

python ./main.py --model ResNet --epoch 30 --batch_size 64 --lr_scheduler --train --save_best --data_path "/home/alexchartrand91" --use_cache