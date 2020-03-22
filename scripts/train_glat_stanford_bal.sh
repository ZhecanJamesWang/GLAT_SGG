#!/usr/bin/env bash
python models/train_rels_glat_universal.py -m sgcls -model_s_m stanford -p 400 \
-tb_log_dir summaries/stanford_glat_sgcls_bal \
-lr 1e-4 \
-ngpu 1 \
-ckpt checkpoints/stanford_bal/vgrel-7.tar \
-save_dir checkpoints/stanford_glat_sgcls_bal \
-adam \
-b 20 \
| tee ./logs/train_stanford_glat_sgcls_bal.txt \

