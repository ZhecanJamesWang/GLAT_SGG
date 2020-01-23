#!/usr/bin/env bash
python models/train_rels_stanford.py -m sgcls -p 400 \
-tb_log_dir summaries/stanford_glat_1 \
-lr 1e-4 \
-ngpu 1 \
-ckpt checkpoints/stanford_1/vgrel-7.tar \
-save_dir checkpoints/stanford_glat_2 \
-adam \
-b 1 \
-nepoch 50 \
