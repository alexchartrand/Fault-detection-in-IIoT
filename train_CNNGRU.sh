#!/usr/bin/env sh

python ./main.py --model CNN-GRU --epoch 30 --batch_size 16 --lr_scheduler --clip --train --save_best --data_path "/home/alexchartrand91" --use_cache