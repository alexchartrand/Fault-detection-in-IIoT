#!/usr/bin/env sh

python ./main.py --model Encoder --epoch 30 --batch_size 12 --lr_scheduler --train --save_best --data_path "/home/alexchartrand91" --use_cache