#!/usr/bin/env bash
python models/train_rels_baseline_1.py -m predcls -p 1000 -clip 5 \
-tb_log_dir summaries/kern_glat_exp3 \
-save_dir checkpoints/kern_glat_exp3 \
-ckpt checkpoints/exp_2_23.tar \
-val_size 5000 \
-adam \
-b 1 \
-lr 1e-4 \
