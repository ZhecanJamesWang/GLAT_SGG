#!/usr/bin/env bash
python models/stanford_bal.py -m sgcls -b 4 -p 400 -lr 1e-4 -ngpu 1 -ckpt checkpoints/vgdet/vg-faster-rcnn.tar -save_dir checkpoints/stanford_bal -adam -rel_knowledge prior_matrices/pred_counts.pkl | tee ./logs/train_stanford_bal.txt

