#!/usr/bin/env bash
python models/train_rels_stanford.py -m predcls -p 400 \
-tb_log_dir summaries/stanford_glat_predcls \
-lr 1e-4 \
-ngpu 1 \
-ckpt checkpoints/stanford_1/vgrel-7.tar \
-save_dir checkpoints/stanford_glat_predcls \
-adam \
-b 1 \
-nepoch 50 \
