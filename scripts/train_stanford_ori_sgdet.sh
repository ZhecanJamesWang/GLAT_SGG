#!/usr/bin/env bash
#python models/stanford_copy.py -m sgcls -b 4 -p 400 -lr 1e-4 -ngpu 1 -ckpt checkpoints/vgdet/vg-faster-rcnn.tar -save_dir checkpoints/stanford_1 -adam

python models/stanford_copy.py -m sgdet -b 6 -p 100 -lr 1e-4 -ngpu 1 -clip 5 \
-ckpt checkpoints/stanford_sgcls/vgrel-7.tar -save_dir checkpoints/stanford_sgdet | tee ./logs/train_stanford_ori_sgdet.txt